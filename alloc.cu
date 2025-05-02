// ---------------------------------------------------------------------------
// [...]
// ---------------------------------------------------------------------------
// Example build & test command:
// nvcc --gpu-architecture=native -Xcompiler -fPIC -shared alloc.cu -o alloc.so -lcuda -lnvrtc && python test.py
// ---------------------------------------------------------------------------
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <chrono>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <ctime>

// ---------------------------------------------------------------------------
// Global configuration variables
// ---------------------------------------------------------------------------
static int g_num_devices = -1; // init on 1st call to get_device_context()
static int g_prealloc_extra_required = 1, g_prealloc_extra_alloc = 10; // configured by set_prealloc_config()
static size_t g_custom_alloc_threshold = 8ULL * 1024ULL * 1024ULL; // use custom allocator above this
static size_t g_free_mapped_start_threshold = 16ULL * 1024ULL * 1024ULL * 1024ULL; // auto unmap on malloc above this
static size_t g_free_mapped_end_threshold   = 2ULL  * 1024ULL * 1024ULL * 1024ULL; // stop auto unmapping at this point
static std::string g_sass_filename; // If empty, don't output SASS. Set via set_output_sass().

// ---------------------------------------------------------------------------
// Compile Time Settings
// ---------------------------------------------------------------------------
//#define DEBUG_PRINTF
constexpr bool   ALWAYS_OUTPUT_SASS = true;    // always output assembly to "sass" file (even if filename is empty)
constexpr int    MAX_SM = 200;                 // enough for all NVIDIA GPUs up to GB300 (but not e.g. MI300X!)
constexpr int    FORCED_HASH = 0;              // 0xB3000 for H100, 0xAB000 for GH200 96GiB, 0x1EF000 for GB200
constexpr int    L2_SIDE_TEST_ITERATIONS = 25; // increases warmup time (at init & per page) but improves accuracy
constexpr bool   TRY_CUDA_FREE_ON_MISS = true; // cudaFreeAsync for unknown pointers (e.g. if alloc threshold changes)

constexpr size_t CHUNK_SIZE = 4096;            // 4KiB (= granularity of side switch on H100/GB200)
constexpr size_t PAGE_SIZE  = 2 * 1024 * 1024; // 2MiB (= granularity of NVIDIA MMU pages since Pascal)
constexpr size_t UNBACKED_VIRTUAL_PAGES = 2048ULL * 1024ULL * 1024ULL / PAGE_SIZE; // maximum we allocate in one go

// Offsets into the cpu_side_info array
constexpr int OFFSET_ZEROED_COUNTER    = MAX_SM;
constexpr int OFFSET_AVERAGE_LATENCY   = MAX_SM + 1;
constexpr int OFFSET_SIDE_HASH_MASK    = MAX_SM + 2;
constexpr int OFFSET_NUM_SM_SIDE0      = MAX_SM + 3;
constexpr int OFFSET_NUM_SM_SIDE1      = MAX_SM + 4;
constexpr int OFFSET_MIN_SM_PER_SIDE   = MAX_SM + 5;

// ---------------------------------------------------------------------------
// Everything needed for side-aware page-based allocation
// ---------------------------------------------------------------------------
struct LargeAllocInfo {
    size_t user_requested   = 0;
    void*  base_ptr         = nullptr;  // Points to the correctly aligned section for mapping
    void*  alloc_ptr        = nullptr;  // Original pointer from cuMemAddressReserve
    size_t aligned_size     = 0;
    bool   use_compression  = false;
    cudaStream_t last_use_stream = 0;
    std::vector<CUmemGenericAllocationHandle> handles; // One handle per 2MiB page
    // Track which side was actually used for each page so we know where to return it:
    std::vector<int> side_used; // side (0 or 1) used for each page (returns to that pool on free)
};

// ---------------------------------------------------------------------------
// Device Context structure to track per-device data
// ---------------------------------------------------------------------------
typedef struct {
    unsigned char side_index[MAX_SM]; // struct to pass this as a single argument to the GPU
} ParamSmSide;

struct DeviceContext {
    // Device properties
    bool initialized = false;
    int device_id = -1;
    int num_sms = 0;

    // CPU memory (used on CPU or passed as kernel parameter)
    int cpu_side_info[MAX_SM * 2];
    ParamSmSide param_sm_side;

    // GPU memory for side info
    unsigned int* gpu_allocator_metadata = nullptr;
    unsigned int* gpu_side_info = nullptr;
    unsigned char* gpu_side_index = nullptr;
    unsigned char* gpu_scratch_buffer = nullptr;

    // Unbacked virtual memory tracking
    unsigned char cpu_page_side[UNBACKED_VIRTUAL_PAGES];

    // Memory pools
    std::vector<CUmemGenericAllocationHandle> free_handles_side0;
    std::vector<CUmemGenericAllocationHandle> free_handles_side1;
    std::unordered_map<size_t, std::vector<LargeAllocInfo>> large_alloc_cache;
    std::unordered_map<void*, LargeAllocInfo> large_alloc_registry;

    // State tracking
    size_t total_mapped_free = 0;
    bool cross_side_warning_issued = false;
    bool compression_available = false;
    std::unordered_map<size_t, int> size_side_map;

    // Initialize the context for a specific device
    void initialize(cudaStream_t stream);

    // -------------------------------------------------------------------
    // NVRTC kernel caching (indexed by header ID)
    // -------------------------------------------------------------------
    struct KernelCacheEntry {
        CUmodule module = nullptr;                    // compiled module for header
        CUfunction funcs[4] = {nullptr,nullptr,nullptr,nullptr};
    };

    std::vector<KernelCacheEntry> kernel_cache;       // index = header ID (0 default)
    std::vector<std::string>      header_strings;     // same index – header source
    std::unordered_map<std::string,int> header_to_id; // reverse lookup
};

// ---------------------------------------------------------------------------
// Helper Functions & Classes
// ---------------------------------------------------------------------------
template<class T>
constexpr inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

template<class... Ts>
__device__ __host__ static void debugf(const char* fmt, Ts... args) {
#ifdef DEBUG_PRINTF
    std::printf(fmt, args...);
#endif
}

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

template <typename T>
inline void __checkCudaErrors(T err, const char *file, const int line) {
    if (err != 0) {
        const char *errorStr = "";
        if constexpr (std::is_same<T, cudaError_t>::value) { errorStr = cudaGetErrorString(err); }
        else if constexpr (std::is_same<T, CUresult>::value) { cuGetErrorString(err, &errorStr); }
        else if constexpr (std::is_same<T, nvrtcResult>::value) { errorStr = nvrtcGetErrorString(err); }
        fprintf(stderr, "checkCudaErrors() error = %04d \"%s\" from file <%s>, line %d.\n", err, errorStr, file, line);
        assert(false);
    }
}
#endif

// Compiles a CUDA source file (.cu) to CUBIN using NVRTC (asserts on failure)
static void compileFileToCUBIN(CUdevice device, char **cubin_out, const char *filename,
                              const char *header_code = nullptr, size_t *cubin_size_out = nullptr,
                              const char *include_path = "/usr/local/cuda/include/") {
    assert(filename && *filename && "NVRTC requires a non-empty filename.");

    std::ifstream file_stream(filename, std::ios::binary);
    assert(file_stream && "Cannot open NVRTC source file.");

    std::string source_code((std::istreambuf_iterator<char>(file_stream)), {});
    std::string final_source = header_code ? (std::string(header_code) + "\n" + source_code) : source_code;

    int major_cc, minor_cc; // this won't include optional 'a' suffix for e.g. H100 sm_90a
    checkCudaErrors(cuDeviceGetAttribute(&major_cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&minor_cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    std::string arch_flag = "--gpu-architecture=sm_" + std::to_string(major_cc) + std::to_string(minor_cc);

    std::vector<std::string> opts_str = {"--generate-line-info", "-use_fast_math",
                                         arch_flag, std::string("-I") + include_path};
    std::vector<const char*> opts_c;
    opts_c.reserve(opts_str.size());
    for(const auto& s : opts_str) opts_c.push_back(s.c_str());

    nvrtcProgram program;
    checkCudaErrors(nvrtcCreateProgram(&program, final_source.c_str(), filename, 0, nullptr, nullptr));
    nvrtcResult compile_res = nvrtcCompileProgram(program, opts_c.size(), opts_c.data());

    size_t log_size = 0;
    checkCudaErrors(nvrtcGetProgramLogSize(program, &log_size));
    if (log_size > 1) { // Print log even on success (for warnings)
        std::string log(log_size, '\0');
        checkCudaErrors(nvrtcGetProgramLog(program, &log[0]));
        std::cerr << "NVRTC Log (" << filename << "):\n" << log << std::endl;
    }

    if (compile_res != NVRTC_SUCCESS) {
        const char* error_string = nvrtcGetErrorString(compile_res);
        std::cerr << "Error: NVRTC compilation failed for '" << filename
                  << "' with error: " << (error_string ? error_string : "Unknown NVRTC error")
                  << " (Code: " << compile_res << "). Check NVRTC log above for details.\n";
        checkCudaErrors(nvrtcDestroyProgram(&program));
        assert(false);
    }

    size_t cubin_size = 0;
    checkCudaErrors(nvrtcGetCUBINSize(program, &cubin_size));
    if (cubin_size_out) *cubin_size_out = cubin_size;

    *cubin_out = static_cast<char*>(malloc(cubin_size));
    assert(*cubin_out && "Failed malloc for CUBIN");

    checkCudaErrors(nvrtcGetCUBIN(program, *cubin_out));
    checkCudaErrors(nvrtcDestroyProgram(&program));

    // Extract SASS from the generated CUBIN and append it to the output file
    if (ALWAYS_OUTPUT_SASS || !g_sass_filename.empty()) {
        char tmpPath[] = "tmp_cubin";
        FILE* tmpFile = fopen(tmpPath, "wb");
        if (tmpFile) {
            fwrite(*cubin_out, 1, cubin_size, tmpFile);
            fclose(tmpFile);

            std::string cmd = "cuobjdump --dump-sass " + std::string(tmpPath) + " 2>/dev/null";
            if (FILE* pipe = popen(cmd.c_str(), "r")) {
                std::ofstream sass_file(g_sass_filename.empty() ? "sass" : g_sass_filename, std::ios::app);
                if (sass_file.is_open()) {
                    char buffer[4096];
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    std::strftime(buffer, sizeof(buffer), "%F %T", std::localtime(&now_c));

                    sass_file << "==================== Start SASS dump (" << buffer << ") =====================\n";
                    while (fgets(buffer, sizeof(buffer), pipe)) {
                        sass_file << buffer;
                    }
                    sass_file << "===================== End SASS dump ======================\n";
                }
                pclose(pipe);
            }
        }
        remove(tmpPath);
    }
}

static CUmodule loadCUBIN(char *cubin, CUdevice cuDevice, bool free_cubin=true) {
    CUmodule module;
    checkCudaErrors(cuModuleLoadData(&module, cubin));
    if (free_cubin) free(cubin);
    return module;
}

// Multi-GPU only: save current device and restore it when destroyed (out of scope)
class ScopedSetDevice {
public:
    explicit ScopedSetDevice(int new_device) {
        if (g_num_devices != 1) {
            cudaGetDevice(&old_device);
            cudaSetDevice(new_device);
        }
    }
    ~ScopedSetDevice() {
        if (g_num_devices != 1) {
            cudaSetDevice(old_device);
        }
    }
private:
    int old_device;
};

// ---------------------------------------------------------------------------
// Device functions & kernels for side info
// ---------------------------------------------------------------------------
__device__ __forceinline__ int test_latency_l2(unsigned int* data, size_t offset) {
    unsigned int old_value = atomicExch(&data[offset], 0); // also warms up the cache!
    long long int start_clock = clock64();
    for (int i = 0; i < L2_SIDE_TEST_ITERATIONS; i++) {
        int value = atomicInc(&data[offset], 99999999);
        offset += (value > L2_SIDE_TEST_ITERATIONS*10) ? 1 : 0;
    }
    int latency = clock64() - start_clock;
    data[offset] = old_value;
    return latency;
}

__global__ void init_side_info(unsigned int* base_page, unsigned int *side_info, unsigned char *side_index) {
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

        int offset = 4 * smid;
        assert(offset * sizeof(int) < CHUNK_SIZE);

        __nanosleep(((smid+4)% 16) * 2000 + 100);
        int total_latency = test_latency_l2(base_page, offset);
        side_info[smid] = total_latency;
        atomicAdd(&side_info[OFFSET_AVERAGE_LATENCY], total_latency);

        int num_done = atomicInc(&side_info[OFFSET_ZEROED_COUNTER], gridDim.x - 1);
        if (num_done == gridDim.x - 1) {
            int average_latency = side_info[OFFSET_AVERAGE_LATENCY] / gridDim.x;
            debugf("Average L2-latency threshold: %.1f\n", (float)average_latency / (float)L2_SIDE_TEST_ITERATIONS);

            // SM0 is always side 0 (everything else is relative to it)
            int far_side =  (side_info[0] > average_latency) ? 0 : 1;
            int near_side = (side_info[0] > average_latency) ? 1 : 0;

            int side0_counter = 0, side1_counter = 0;
            for (int i = 0; i < gridDim.x; i++) {
                int latency = side_info[i];
                side_info[i] = (latency > average_latency) ? far_side : near_side;
                if ((side_info[i] & 1) == 0) {
                    side_info[i] |= (side0_counter++) << 1;
                } else {
                    side_info[i] |= (side1_counter++) << 1;
                }
                side_index[i] = (unsigned char)side_info[i];
                debugf("[SM %3d] L2-latency = %.1f -> side=%d idx=%d\n",
                       i, (float)latency / (float)L2_SIDE_TEST_ITERATIONS,
                       (side_info[i] & 1), (side_info[i] >> 1));
            }
            side_info[OFFSET_AVERAGE_LATENCY] = average_latency;
            side_info[OFFSET_NUM_SM_SIDE0]    = side0_counter;
            side_info[OFFSET_NUM_SM_SIDE1]    = side1_counter;
            side_info[OFFSET_MIN_SM_PER_SIDE] = min(side0_counter, side1_counter);

            if constexpr (FORCED_HASH == 0) {
                unsigned long long int addr_int = reinterpret_cast<unsigned long long int>(base_page);
                if (addr_int % ((size_t)PAGE_SIZE) != 0) {
                    debugf("ERROR: base_page not 2MiB-aligned\n");
                    assert(false);
                    return;
                }
                int base_side = side_info[smid] & 1;
                int check_start_bit = 4;
                int check_last_bit  = 20;
                int toggle_bits = 0;
                for (int i = check_start_bit; i <= check_last_bit; i++) {
                    int bitmask = 1 << i;
                    int offset2 = bitmask / sizeof(int);
                    int total_latency2 = test_latency_l2(base_page, offset2);
                    int offset_side = (total_latency2 > average_latency)? far_side : near_side;
                    if (offset_side != base_side) {
                        toggle_bits |= bitmask;
                    }
                }
                side_info[OFFSET_SIDE_HASH_MASK] = toggle_bits;
                debugf("Detected side-hash bits: 0x%X\n", toggle_bits);
                if (!(toggle_bits & CHUNK_SIZE) || (toggle_bits & (CHUNK_SIZE - 1))) {
                    printf("\nERROR: CHUNK_SIZE %d doesn't work with hash %x\n\n", (int)CHUNK_SIZE, toggle_bits);
                    assert(false);
                }
            } else {
                side_info[OFFSET_SIDE_HASH_MASK] = FORCED_HASH;
            }
        }
    } else if (threadIdx.x >= 32) {
        __nanosleep(10000);
    }
}

__global__ void test_page_latency(unsigned int* ptr, unsigned int *side_info, unsigned char* page_side, int num_pages)
{
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        int near_side = side_info[smid] & 1;
        int far_side  = 1 - near_side;
        int average_latency = side_info[OFFSET_AVERAGE_LATENCY];

        debugf("Testing %u pages.\n", num_pages);
        for (int i = 0; i < num_pages; i++) {
            size_t offset = (size_t)i * (PAGE_SIZE / sizeof(unsigned int));
            int total_latency = test_latency_l2(ptr, offset);
            page_side[i] = (total_latency > average_latency) ? far_side : near_side;
            debugf("[SM %3d] Page %3d: L2-latency = %.1f (raw: %u) -> side=%d\n",
                   smid, i, (float)total_latency / (float)L2_SIDE_TEST_ITERATIONS, total_latency, page_side[i]);
        }
    }
}

// Device contexts array - indexed by device ID
static std::vector<DeviceContext> g_deviceContexts;

static DeviceContext& get_device_context(int device=-1) {
    if (g_num_devices < 0) {
        cudaGetDeviceCount(&g_num_devices);
        g_deviceContexts.resize(g_num_devices);
    }

    if (device < 0) {
        if (g_num_devices == 1) {
            device = 0;
        } else {
            cudaGetDevice(&device);
        }
    }

    if (device >= g_deviceContexts.size()) {
        assert(device < g_num_devices);
        g_deviceContexts.resize(g_num_devices);
    }

    DeviceContext& ctx = g_deviceContexts[device];
    if (!ctx.initialized) {
        ctx.device_id = device;
        ctx.initialize(0);
    }
    return ctx;
}

// NVRTC kernel compilation helper (for side_aware_memcpy)
// TODO: this isn't actually memcpy anymore
static CUfunction getMemcpyKernel(DeviceContext &ctx, int header_id, bool use_64bit) {
    assert(header_id < ctx.kernel_cache.size() && header_id < ctx.header_strings.size());
    DeviceContext::KernelCacheEntry &entry = ctx.kernel_cache[header_id];

    int idx = use_64bit ? 1 : 0; // select variant
    if (entry.funcs[idx] == nullptr) {
        assert(entry.module == nullptr);

        CUdevice cuDevice;
        cuCtxGetDevice(&cuDevice);

        char *cubin = nullptr;
        const std::string& s = ctx.header_strings[header_id];
        const char* header_ptr = s.empty() ? nullptr : s.c_str();
        compileFileToCUBIN(cuDevice, &cubin, "sideaware_kernels.cu", header_ptr);
        entry.module = loadCUBIN(cubin, cuDevice);

        const char* fn_names[2] = { "side_aware_memcpy_32", "side_aware_memcpy_64" };
        for (int i = 0; i < 2; i++) {
            CUfunction ftmp;
            CUresult rc = cuModuleGetFunction(&ftmp, entry.module, fn_names[i]);
            if (rc != CUDA_SUCCESS) {
                const char *errStr = nullptr; cuGetErrorString(rc,&errStr);
                std::cerr << "Failed to get " << fn_names[i] << " : " << (errStr?errStr:"") << std::endl;
                std::abort();
            }
            entry.funcs[i] = ftmp;
        }
    }
    return entry.funcs[idx];
}

// Query the device allocation constraints
static void init_allocation_constraints(DeviceContext& ctx)
{
    int comp_available;
    cuDeviceGetAttribute(&comp_available, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, ctx.device_id);
    ctx.compression_available = (comp_available != 0);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ctx.device_id;

    size_t granularity;
    CUresult res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(res == CUDA_SUCCESS && granularity == PAGE_SIZE); // Verify granularity equals PAGE_SIZE

    if (ctx.compression_available) {
        prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        assert(res == CUDA_SUCCESS && granularity == PAGE_SIZE);
    }
}

static CUmemAllocationProp get_allocation_constraints(DeviceContext& ctx, bool use_compression=false)
{
    use_compression &= ctx.compression_available;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ctx.device_id;
    prop.allocFlags.compressionType = (use_compression) ? CU_MEM_ALLOCATION_COMP_GENERIC : 0;
    return prop;
}

void DeviceContext::initialize(cudaStream_t stream) {
    if (initialized) return;
    ScopedSetDevice guard(device_id);

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

    init_allocation_constraints(*this);
    cudaMalloc(&gpu_allocator_metadata, PAGE_SIZE);
    assert((uintptr_t)gpu_allocator_metadata % PAGE_SIZE == 0);
    gpu_side_info = &gpu_allocator_metadata[16 * 1024];
    gpu_side_index = (unsigned char*)&gpu_allocator_metadata[32 * 1024];
    gpu_scratch_buffer = (unsigned char*)&gpu_allocator_metadata[128 * 1024];

    init_side_info<<<num_sms, 512, 0, stream>>>(gpu_allocator_metadata, gpu_side_info, gpu_side_index);

    unsigned char cpu_side_index[MAX_SM];
    cudaMemcpyAsync(cpu_side_index, gpu_side_index, MAX_SM * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    cudaMemcpy(cpu_side_info, gpu_side_info, MAX_SM * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost); // syncs both

    for (int i = 0; i < MAX_SM; i++) {
        param_sm_side.side_index[i] = cpu_side_index[i];
    }
    header_strings.resize(1);
    kernel_cache.resize(1);

    initialized = true;
}

// ---------------------------------------------------------------------------
// Unmap free blocks if threshold is exceeded
// ---------------------------------------------------------------------------
static void unmapFreeLargeAllocations(DeviceContext& ctx, size_t start_threshold=0, size_t end_threshold=0) {
    if (ctx.total_mapped_free <= start_threshold) return;

    ScopedSetDevice guard(ctx.device_id);
    cudaDeviceSynchronize();

    for (auto &kv : ctx.large_alloc_cache) {
        auto &vec = kv.second;
        auto it = vec.begin();

        while (it != vec.end() && ctx.total_mapped_free > end_threshold) {
            LargeAllocInfo &blk = *it;
            size_t size = blk.aligned_size;
            size_t num_pages = blk.aligned_size / PAGE_SIZE;
            CUdeviceptr base = (CUdeviceptr)blk.base_ptr;

            for (size_t i = 0; i < num_pages; i++) {
                cuMemUnmap(base + i * PAGE_SIZE, PAGE_SIZE);
                // Return handle to the side it was allocated from:
                int side = blk.side_used[i];
                if (side == 0) {
                    ctx.free_handles_side0.push_back(blk.handles[i]);
                } else {
                    ctx.free_handles_side1.push_back(blk.handles[i]);
                }
            }
            // Use alloc_ptr for freeing address space, not base_ptr
            cuMemAddressFree((CUdeviceptr)blk.alloc_ptr, blk.aligned_size + 2 * PAGE_SIZE);

            assert(ctx.total_mapped_free >= size);
            ctx.total_mapped_free -= size;
            it = vec.erase(it);
        }
        if (ctx.total_mapped_free <= end_threshold)
            break;
    }
    cudaDeviceSynchronize();
}

static size_t release_unused_memory_device(int device) {
    DeviceContext& ctx = get_device_context(device);
    ScopedSetDevice guard(device);

    // Unmap all cached blocks then release handles from side0/side1 pools
    unmapFreeLargeAllocations(ctx);

    size_t freed_memory = 0;
    for (auto &h : ctx.free_handles_side0) {
        cuMemRelease(h);
        freed_memory += PAGE_SIZE;
    }
    ctx.free_handles_side0.clear();

    for (auto &h : ctx.free_handles_side1) {
        cuMemRelease(h);
        freed_memory += PAGE_SIZE;
    }
    ctx.free_handles_side1.clear();

    cudaDeviceSynchronize();
    return freed_memory;
}

// ---------------------------------------------------------------------------
// Pre-allocate new physical handles (=2MiB pages of GPU physical memory)
// Called when we run out but allocates some extra to reduce future calls
// This is useful because this requires fully synchronous operations
// ---------------------------------------------------------------------------
static CUresult preAllocateHandles(DeviceContext& ctx, int countNeeded, bool useCompression, cudaStream_t stream)
{
    if (countNeeded == 0) return CUDA_SUCCESS;

    CUmemAllocationProp prop = get_allocation_constraints(ctx, useCompression);
    CUresult lastError = CUDA_SUCCESS;
    int totalAllocated = 0;

    // We'll do it in batches so we don't exceed UNBACKED_VIRTUAL_PAGES at once.
    while (countNeeded > 0) {
        int batch = std::min<int>(countNeeded, UNBACKED_VIRTUAL_PAGES);
        int batchAllocated = 0;

        // Reserve a *temporary* VA space for up to 'batch' pages (freed before returning)
        CUdeviceptr dptr;
        lastError = cuMemAddressReserve(&dptr, batch * PAGE_SIZE, 0, 0, 0);
        if (lastError != CUDA_SUCCESS) {
            break; // Can't continue this batch, but process what we have so far
        }

        // Create + map each handle:
        std::vector<CUmemGenericAllocationHandle> tempHandles(batch);
        for (int i = 0; i < batch; i++) {
            lastError = cuMemCreate(&tempHandles[i], PAGE_SIZE, &prop, 0);
            if (lastError != CUDA_SUCCESS) {
                tempHandles.resize(i);
                batch = i; // adjust batch size
                break;
            }

            lastError = cuMemMap(dptr + i * PAGE_SIZE, PAGE_SIZE, 0, tempHandles[i], 0);
            if (lastError != CUDA_SUCCESS) {
                cuMemRelease(tempHandles[i]);
                tempHandles.resize(i);
                batch = i;
                break;
            }
            batchAllocated++;
        }

        // If failed to allocate anything in this iteration, free the VA space and continue
        if (batchAllocated == 0) {
            cuMemAddressFree(dptr, batch * PAGE_SIZE);
            break;
        }

        // Set read/write access for everything we allocated (in the VA space)
        CUmemAccessDesc accessDesc;
        accessDesc.location.id   = prop.location.id;
        accessDesc.location.type = prop.location.type;
        accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        lastError = cuMemSetAccess(dptr, batchAllocated * PAGE_SIZE, &accessDesc, 1);
        if (lastError != CUDA_SUCCESS) {
            for (int i = 0; i < batchAllocated; i++) {
                cuMemUnmap(dptr + i * PAGE_SIZE, PAGE_SIZE);
                cuMemRelease(tempHandles[i]);
            }
            cuMemAddressFree(dptr, batch * PAGE_SIZE);
            break;
        }

        // Zero out memory before using it (TODO: do we need this, or is it overkill?)
        cudaDeviceSynchronize();
        lastError = cuMemsetD8(dptr, 0, batchAllocated * PAGE_SIZE);
        if (lastError != CUDA_SUCCESS) {
            printf("ERROR: Failed to zero out memory before test (this should never happen?!)\n");
            assert(false);
            return lastError;
        }

        // Classify each page (side 0 or 1) by calling test_page_latency
        test_page_latency<<<1, 512, 0, stream>>>(
            reinterpret_cast<unsigned int*>(dptr), ctx.gpu_side_info, ctx.gpu_scratch_buffer, batchAllocated);
        cudaMemcpy(ctx.cpu_page_side, ctx.gpu_scratch_buffer, batchAllocated, cudaMemcpyDeviceToHost);

        // Unmap + place each handle in the correct side's pool
        for (int i = 0; i < batchAllocated; i++) {
            cuMemUnmap(dptr + i * PAGE_SIZE, PAGE_SIZE);
            int side = ctx.cpu_page_side[i];
            if (side == 0) {
                ctx.free_handles_side0.push_back(tempHandles[i]);
            } else {
                ctx.free_handles_side1.push_back(tempHandles[i]);
            }
        }
        cuMemAddressFree(dptr, batch * PAGE_SIZE);

        totalAllocated += batchAllocated;
        countNeeded -= batchAllocated;

        if (batchAllocated < batch) {
            break; // probably out of GPU memory?
        }
    }

    // If we allocated anything at all, consider it at least a partial success
    if (totalAllocated > 0) {
        return CUDA_SUCCESS;
    }
    return lastError;
}

static CUresult ensureFreeHandlesAvailable(DeviceContext& ctx, size_t needed, bool useCompression, cudaStream_t stream)
{
    size_t needed_per_side = (needed + g_prealloc_extra_required) / 2;
    size_t free0 = ctx.free_handles_side0.size();
    size_t free1 = ctx.free_handles_side1.size();
    size_t needed_side0 = (needed_per_side > free0) ? (needed_per_side - free0) : 0;
    size_t needed_side1 = (needed_per_side > free1) ? (needed_per_side - free1) : 0;
    size_t needed_worst_case = max(needed_side0, needed_side1);
    size_t needed_both_sides = 2 * needed_worst_case;

    if (needed_both_sides > 0) {
        needed_both_sides += g_prealloc_extra_alloc;
        CUresult rc = preAllocateHandles(ctx, needed_both_sides, useCompression, stream);
        return rc;
    }
    return CUDA_SUCCESS;
}

constexpr inline int pickSideFromVA(uint64_t va)
{
    return (int)((va >> 21) & 1ULL);
}

static CUresult allocateCompressible(DeviceContext& ctx, LargeAllocInfo &info, size_t size, bool use_compression=false)
{
    info.aligned_size = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    info.use_compression = (use_compression && ctx.compression_available);

    // 1) Pre-check we have enough free handles, otherwise allocate more:
    size_t num_pages = info.aligned_size / PAGE_SIZE;
    CUresult rc = ensureFreeHandlesAvailable(ctx, num_pages, info.use_compression, info.last_use_stream);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    // Determine the desired "start side" for this allocation size
    // If we haven't seen this size before, choose the side with fewer handles
    if (ctx.size_side_map.find(info.aligned_size) == ctx.size_side_map.end()) {
        int side0_free = ctx.free_handles_side0.size();
        int side1_free = ctx.free_handles_side1.size();
        ctx.size_side_map[info.aligned_size] = (side0_free >= side1_free) ? 0 : 1;
    }
    int desiredStartSide = ctx.size_side_map[info.aligned_size];

    // 2) Reserve VA space with EXTRA space (extra page = 2MiB)
    // This ensures we'll find a properly-aligned section with the desired side
    CUdeviceptr allocPtr = 0;
    size_t extraSize = info.aligned_size + PAGE_SIZE;
    rc = cuMemAddressReserve(&allocPtr, extraSize, 0, 0, 0);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    info.alloc_ptr = reinterpret_cast<void*>(allocPtr);
    CUdeviceptr basePtr = allocPtr;

    // If the first page is not on the desired side, try the second page
    int firstPageSide = pickSideFromVA((uint64_t)basePtr);
    if (firstPageSide != desiredStartSide) {
        basePtr += PAGE_SIZE; // Skip to next 2MiB page
    }

    // Verify that this page has the desired side
    int basePageSide = pickSideFromVA((uint64_t)basePtr);
    if (basePageSide != desiredStartSide) {
        // This should never happen with our hash function using bit 21?!
        printf("ERROR: Failed to find desired side in 2 consecutive 2MiB pages\n");
        cuMemAddressFree(allocPtr, extraSize);
        return CUDA_ERROR_UNKNOWN;
    }

    // Set the basePtr to point to the section with the desired side
    info.base_ptr = reinterpret_cast<void*>(basePtr);
    info.handles.resize(num_pages);
    info.side_used.resize(num_pages);

    // 3) Map each page from whichever side is indicated by bit 21
    for (size_t i = 0; i < num_pages; i++) {
        uint64_t thisVa = (uint64_t)(basePtr + i * PAGE_SIZE);
        int desiredSide = pickSideFromVA(thisVa);

        // pop from the correct side if available
        std::vector<CUmemGenericAllocationHandle>* correctPool =
            (desiredSide == 0) ? &ctx.free_handles_side0 : &ctx.free_handles_side1;
        std::vector<CUmemGenericAllocationHandle>* otherPool   =
            (desiredSide == 0) ? &ctx.free_handles_side1 : &ctx.free_handles_side0;

        if (!correctPool->empty()) {
            info.handles[i] = correctPool->back();
            correctPool->pop_back();
            info.side_used[i] = desiredSide;
        } else {
            // Try to fallback to the other side
            if (otherPool->empty()) {
                // We've run out of GPU memory on both sides
                cuMemAddressFree(allocPtr, extraSize);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            // Warn the first time this happens
            if (!ctx.cross_side_warning_issued) {
                printf("WARNING: Cross-side handle usage on device %d!\n", ctx.device_id);
                ctx.cross_side_warning_issued = true;
            }
            info.handles[i] = otherPool->back();
            otherPool->pop_back();
            info.side_used[i] = desiredSide;
        }

        rc = cuMemMap(basePtr + i * PAGE_SIZE, PAGE_SIZE, 0, info.handles[i], 0);
        if (rc != CUDA_SUCCESS) {
            // Return the handle to whichever side we took it from
            if (info.side_used[i] == 0) {
                ctx.free_handles_side0.push_back(info.handles[i]);
            } else {
                ctx.free_handles_side1.push_back(info.handles[i]);
            }
            // Cleanup on allocation failure
            for (size_t j = 0; j < i; j++) {
                cuMemUnmap(basePtr + j * PAGE_SIZE, PAGE_SIZE);
                if (info.side_used[j] == 0) {
                    ctx.free_handles_side0.push_back(info.handles[j]);
                } else {
                    ctx.free_handles_side1.push_back(info.handles[j]);
                }
            }
            cuMemAddressFree(allocPtr, extraSize);
            return rc;
        }
    }

    // 4) Enable read/write access for the entire allocation in the VA space
    CUmemAccessDesc accessDesc;
    accessDesc.location.id   = ctx.device_id;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    rc = cuMemSetAccess(basePtr, info.aligned_size, &accessDesc, 1);
    if (rc != CUDA_SUCCESS) {
        // Cleanup on allocation failure
        for (size_t i = 0; i < num_pages; i++) {
            cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
            if (info.side_used[i] == 0) ctx.free_handles_side0.push_back(info.handles[i]);
            else                       ctx.free_handles_side1.push_back(info.handles[i]);
        }
        cuMemAddressFree(allocPtr, extraSize);
        return rc;
    }

    // 5) Zero memory to be safe (TODO: make this configurable)
    cudaDeviceSynchronize();
    rc = cuMemsetD8(basePtr, 0, info.aligned_size);
    if (rc != CUDA_SUCCESS) {
        // Cleanup on allocation failure
        for (size_t i = 0; i < num_pages; i++) {
            cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
            if (info.side_used[i] == 0) ctx.free_handles_side0.push_back(info.handles[i]);
            else                       ctx.free_handles_side1.push_back(info.handles[i]);
        }
        cuMemAddressFree(allocPtr, extraSize);
        return rc;
    }

    return CUDA_SUCCESS;
}

static void* sideaware_reuse_alloc(DeviceContext& ctx, size_t alignedSize, cudaStream_t currentStream)
{
    auto &vec = ctx.large_alloc_cache[alignedSize];
    if (vec.empty()) return nullptr;

    LargeAllocInfo info = vec.back();
    assert(ctx.total_mapped_free >= info.aligned_size);
    ctx.total_mapped_free -= info.aligned_size;
    vec.pop_back();

    // If it's a different stream, we might need to synchronize
    // TODO: is this safe or do we need a full device synchronization?
    if (info.last_use_stream != currentStream) {
        cudaStreamSynchronize(info.last_use_stream);
        info.last_use_stream = currentStream;
    }

    // Insert back into active registry
    ctx.large_alloc_registry[info.base_ptr] = info;
    return info.base_ptr;
}

static void* sideaware_new_alloc(DeviceContext& ctx, size_t userSize, cudaStream_t stream)
{
    LargeAllocInfo info;
    info.user_requested = userSize;
    info.last_use_stream = stream;
    CUresult rc = allocateCompressible(ctx, info, userSize);
    if (rc != CUDA_SUCCESS)
        return nullptr;
    ctx.large_alloc_registry[info.base_ptr] = info;
    return info.base_ptr;
}

static void* sideaware_malloc_large(DeviceContext& ctx, size_t size, cudaStream_t stream)
{
    size_t alignedSize = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

    // 1) Try exact-size reuse
    void* p = sideaware_reuse_alloc(ctx, alignedSize, stream);
    if (p) return p;

    // 2) Unmap free pages if over threshold & try again
    unmapFreeLargeAllocations(ctx, g_free_mapped_start_threshold, g_free_mapped_end_threshold);
    p = sideaware_new_alloc(ctx, size, stream);
    if (p) return p;

    // 3) Try fully releasing all unused memory & try again
    release_unused_memory_device(ctx.device_id);
    return sideaware_new_alloc(ctx, size, stream);
}

static void sideaware_free_large(DeviceContext& ctx, void* ptr, cudaStream_t stream)
{
    if (!ptr) return;
    auto it = ctx.large_alloc_registry.find(ptr);
    if (it == ctx.large_alloc_registry.end()) {
        if constexpr (TRY_CUDA_FREE_ON_MISS) {
            cudaFreeAsync(ptr, stream);
        } else {
            printf("ERROR: Failed to find large allocation to free on device %d\n", ctx.device_id);
            assert(false);
        }
        return;
    }

    LargeAllocInfo info = it->second;
    ctx.large_alloc_registry.erase(it);
    ctx.large_alloc_cache[info.aligned_size].push_back(info);
    ctx.total_mapped_free += info.aligned_size;
}

// ---------------------------------------------------------------------------
// PUBLIC API
// ---------------------------------------------------------------------------
extern "C" {

void* sideaware_malloc(size_t size, int device, cudaStream_t stream) {
    debugf("sideaware_malloc(%zu, %d, %p)\n", size, device, stream);
    void* p = nullptr;
    ScopedSetDevice guard(device);
    DeviceContext& ctx = get_device_context(device);

    if (size >= g_custom_alloc_threshold) {
        p = sideaware_malloc_large(ctx, size, stream);
    } else {
        cudaError_t err = cudaMallocAsync(&p, size, stream);
        if (err == cudaSuccess) {
            return p;
        }

        unmapFreeLargeAllocations(ctx);
        err = cudaMallocAsync(&p, size, stream);
        if (err != cudaSuccess) {
            p = nullptr;
        }
    }

    return p;
}

void sideaware_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    debugf("sideaware_free(%p, %zu, %d, %p)\n", ptr, size, device, stream);

    if (!ptr) return;
    DeviceContext& ctx = get_device_context(device);
    ScopedSetDevice guard(device);

    if (size >= g_custom_alloc_threshold) {
        sideaware_free_large(ctx, ptr, stream);
    } else {
        cudaFreeAsync(ptr, stream);
    }
}

size_t sideaware_release_unused() {
    size_t total_freed = 0;
    for (size_t i = 0; i < g_deviceContexts.size(); i++) {
        if (g_deviceContexts[i].initialized) {
            total_freed += release_unused_memory_device(i);
        }
    }
    return total_freed;
}

// ---------------------------------------------------------------------------
// Generic element‑wise kernel launcher. header_id selects runtime‑compiled
// module to use (0 = default memcpy).
// TODO: assumes device == current device, is this always true with PyTorch?
// ---------------------------------------------------------------------------
void sideaware_elementwise(void* dst, const void* src, size_t size, int device, cudaStream_t stream, int header_id) {
    if (size == 0 || dst == nullptr || src == nullptr) return;
    DeviceContext& ctx = get_device_context(device);
    assert(header_id >= 0 && header_id < ctx.kernel_cache.size());

    int sm_per_side = ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
    int hash = ctx.cpu_side_info[OFFSET_SIDE_HASH_MASK] | (1 << 21);

    unsigned int byte_start = (unsigned long long)src & 15;
    unsigned int byte_end = ((unsigned long long)src + size) & 15;
    size_t size_aligned = size;

    if (byte_start) {
        src = (const uint4*)(((const unsigned char*)src) + sizeof(uint4) - byte_start);
        dst = (uint4*)(((unsigned char*)dst) + sizeof(uint4) - byte_start);
        size_aligned -= (sizeof(uint4) - byte_start);
    }
    if (byte_end) {
        size_aligned -= byte_end;
    }

    unsigned int size_aligned_32b = (unsigned int)size_aligned;
    uint4* dst4 = (uint4*)dst;
    const uint4* src4 = (const uint4*)src;


    bool use64 = (size >= 2ULL*1024*1024*1024) || byte_start || byte_end;
    CUfunction kernel = getMemcpyKernel(ctx, header_id, use64);
    CUstream cuStream = reinterpret_cast<CUstream>(stream);

    void* args[8];
    args[0] = &dst4;
    args[1] = &src4;
    args[2] = use64 ? (void*)&size_aligned : (void*)&size_aligned_32b;
    args[3] = &byte_start;
    args[4] = &byte_end;
    args[5] = &hash;
    args[6] = &sm_per_side;
    args[7] = &ctx.param_sm_side;

    cuLaunchKernel(kernel, ctx.num_sms, 1, 1, 256, 4, 1, 0, cuStream, args, nullptr);
}

void sideaware_memcpy(void* dst, const void* src, size_t size, int device, cudaStream_t stream) {
    sideaware_elementwise(dst, src, size, device, stream, 0);
}

// ---------------------------------------------------------------------------
// NVRTC: allow user to inject custom header code for future elementwise ops
// ---------------------------------------------------------------------------
int sideaware_set_custom_header(const char* header) {
    DeviceContext& ctx = get_device_context();

    if (!header || strlen(header)==0) {
        return 0; // default memcpy
    }

    std::string hdr_str(header);
    auto it = ctx.header_to_id.find(hdr_str);
    if (it != ctx.header_to_id.end()) {
        return it->second; // already assigned
    }

    int new_id = ctx.header_strings.size();
    ctx.header_strings.push_back(hdr_str);
    ctx.kernel_cache.emplace_back(); // default‑constructed
    ctx.header_to_id[hdr_str] = new_id;
    return new_id;
}

// ---------------------------------------------------------------------------
// Additional query/utility/config (mostly per device)
// ---------------------------------------------------------------------------
void fill_sm_sides_tensor(unsigned char* gpu_tensor) {
    DeviceContext& ctx = get_device_context();
    cudaError_t err = cudaMemcpy(gpu_tensor, ctx.gpu_side_index, ctx.num_sms, cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);
}

const int* get_sm_side_summary_ptr()
{
    static int s[5];
    DeviceContext& ctx = get_device_context();

    s[0] = ctx.num_sms;
    s[1] = ctx.cpu_side_info[OFFSET_NUM_SM_SIDE0];
    s[2] = ctx.cpu_side_info[OFFSET_NUM_SM_SIDE1];
    s[3] = ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
    s[4] = ctx.cpu_side_info[OFFSET_SIDE_HASH_MASK];
    return s;
}

void set_custom_alloc_threshold(size_t threshold) {
    g_custom_alloc_threshold = threshold;
}

void set_prealloc_config(int extra_required, int extra_alloc) {
    g_prealloc_extra_required = extra_required;
    g_prealloc_extra_alloc = extra_alloc;
}

void set_free_mapped_thresholds(size_t start_threshold, size_t end_threshold) {
    g_free_mapped_start_threshold = start_threshold;
    g_free_mapped_end_threshold = end_threshold;
}

void set_output_sass(const char* filename) {
    g_sass_filename = (filename && filename[0]) ? filename : "";
}

// ---------------------------------------------------------------------------
// Per-parameter query (rarely required)
// ---------------------------------------------------------------------------

int get_num_sms() {
    DeviceContext& ctx = get_device_context();
    return ctx.num_sms;
}

int get_num_sm_side0() {
    DeviceContext& ctx = get_device_context();
    return ctx.cpu_side_info[OFFSET_NUM_SM_SIDE0];
}

int get_num_sm_side1() {
    DeviceContext& ctx = get_device_context();
    return ctx.cpu_side_info[OFFSET_NUM_SM_SIDE1];
}

int get_min_sm_per_side() {
    DeviceContext& ctx = get_device_context();
    return ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
}

int get_hash_mask() {
    DeviceContext& ctx = get_device_context();
    return ctx.cpu_side_info[OFFSET_SIDE_HASH_MASK];
}

} // extern "C"