// ---------------------------------------------------------------------------
// TODO: Documentation...
// ---------------------------------------------------------------------------
// Example build & test command:
// nvcc -arch=native -Xcompiler -fPIC -shared sideaware.cu -o sideaware.so -lcuda -lnvrtc && python test.py
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
#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>

static const char* SIDEAWARE_MEMCPY_HEADER = R"SIDEAWARE_MEMCPY(
typedef int o0;
typedef int i0;

struct unused {}; // optimized away by compiler
typedef unused o1, o2, o3, i1, i2, i3;

__device__ void elementwise_op(size_t element_idx,
                               o0 &out0, o1 &out1,
                               o2 &out2, o3 &out3,
                               const i0 &in0, const i1 &in1,
                               const i2 &in2, const i3 &in3) {
    out0 = in0;
}

constexpr int unrolled = 4; // unrolled loop iterations (increases register pressure especially with multiple inputs)
constexpr bool reverse_order = true; // process end of array 1st (maximise L2 hits with normal->reverse->normal->...)
constexpr bool input_evict[4] = {true, true, true, true}; // do not keep inputs in L2 (more space for outputs)

constexpr bool support_concurrent_kernels = false; // use atomics to dynamically assign SM side index
constexpr bool input_discard[4] = {0}; // danger: discards from L2 *before* data is written to DRAM

)SIDEAWARE_MEMCPY";

static const char* SIDEAWARE_KERNEL_SOURCE = R"SIDEAWARE_KERNEL(
// ----------------------------------------------------------------------------
// L2 Side Aware Multi-Input / Multi-Output Elementwise kernel (16B alignment)
// ----------------------------------------------------------------------------
#include <cuda_runtime.h>

#ifndef KERNEL_NAME
#define KERNEL_NAME sideaware_elementwise
#endif

#ifndef CUSTOM_VECTOR_OP
template<typename size_type>
__device__ void vector_op(size_type idx, int vec_size,
                          o0 *out0, o1 *out1, o2 *out2, o3 *out3,
                          const i0 *in0, const i1 *in1, const i2 *in2, const i3 *in3,
                          int* sideband_memory, int sideband_value) {
    for (int i = 0; i < vec_size; i++) {
        elementwise_op(idx * vec_size + i, out0[i], out1[i], out2[i], out3[i], in0[i], in1[i], in2[i], in3[i]);
    }
}
#endif

#ifndef LAUNCH_BOUNDS
#define LAUNCH_BOUNDS __launch_bounds__(1024, 1)
#endif

#ifndef FORCE_WRONG_SIDE
#define FORCE_WRONG_SIDE false
#endif

#ifndef FORCE_RANDOM_SIDE
#define FORCE_RANDOM_SIDE false
#endif

constexpr size_t vector_bytes = 16;
constexpr bool sideaware_for_o0 = sizeof(o0) > sizeof(i0);
constexpr int vec_size = vector_bytes / (sideaware_for_o0 ? sizeof(o0) : sizeof(i0));

template<typename T, size_t N>
struct packed {
    T data[N];
    __device__ __forceinline__ T& operator[](int index) { return data[index]; }
    __device__ __forceinline__ const T& operator[](int index) const { return data[index]; }
    __device__ __forceinline__ T* operator+() { return data; }
    __device__ __forceinline__ const T* operator+() const { return data; }
};

typedef packed<o0, vec_size> vo0;
typedef packed<o1, vec_size> vo1;
typedef packed<o2, vec_size> vo2;
typedef packed<o3, vec_size> vo3;
typedef packed<i0, vec_size> vi0;
typedef packed<i1, vec_size> vi1;
typedef packed<i2, vec_size> vi2;
typedef packed<i3, vec_size> vi3;

template<bool evict = false, typename T>
__device__ __forceinline__ T load(const T * __restrict__ src) {
    if constexpr (sizeof(T) == 16) {
        int4 data = evict ? __ldcs((const int4*)src) : __ldcg((const int4*)src);
        return *(T*)&data;
    } else if constexpr (sizeof(T) == 8) {
        int2 data = evict ? __ldcs((const int2*)src) : __ldcg((const int2*)src);
        return *(T*)&data;
    } else if constexpr (sizeof(T) == 4) {
        int data = evict ? __ldcs((const int*)src) : __ldcg((const int*)src);
        return *(T*)&data;
    }
    return *src;
}

template<typename T>
__device__ __forceinline__ void store(T* __restrict__ ptr, const T &value) {
    T* aligned_ptr = (T*)__builtin_assume_aligned(ptr, sizeof(T));
    aligned_ptr[0] = value;
}

template<typename T, typename U>
struct is_same { static constexpr bool value = false; };
template<typename T>
struct is_same<T, T> { static constexpr bool value = true; };

__device__ __forceinline__ void discard_inputs(
    const vi0 * input0, const vi1 * input1, const vi2 * input2, const vi3 * input3) {
    if (input_discard[0] && !is_same<i0, unused>::value && (unsigned long long)input0 % 128 == 0)
        asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(input0));
    if (input_discard[1] && !is_same<i1, unused>::value && (unsigned long long)input1 % 128 == 0)
        asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(input1));
    if (input_discard[2] && !is_same<i2, unused>::value && (unsigned long long)input2 % 128 == 0)
        asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(input2));
    if (input_discard[3] && !is_same<i3, unused>::value && (unsigned long long)input3 % 128 == 0)
        asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(input3));
}

typedef struct {
    unsigned char side_index[MAX_SM];
} param_sm_side_t;

extern "C" __global__ LAUNCH_BOUNDS void KERNEL_NAME(
        size_type num_vectors_16B,
        vo0* __restrict__ output0, vo1* __restrict__ output1,
        vo2* __restrict__ output2, vo3* __restrict__ output3,
        const vi0 * __restrict__ input0, const vi1 * __restrict__ input1,
        const vi2 * __restrict__ input2, const vi3 * __restrict__ input3,
        int* mem, int val,
        unsigned int num_sm_per_side,
        unsigned int* zeroed_scratch_buffer,
        const param_sm_side_t params) {
    // ...
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
    unsigned int sm_side = params.side_index[smid] & 1;
    unsigned int sm_side_index;

    if constexpr (support_concurrent_kernels) {
        // Use atomics to dynamically assign SM side index
        __shared__ unsigned int shared_sm_side_index;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            sm_side_index = atomicInc(&zeroed_scratch_buffer[sm_side], 999);
            if (sm_side_index >= num_sm_per_side) {
                // Switch to the 'wrong side' as a failsafe so we still process every element
                sm_side = 1 - sm_side;
                sm_side_index = atomicInc(&zeroed_scratch_buffer[sm_side], 999);
            }
            unsigned int total_both_sides = atomicInc(&zeroed_scratch_buffer[2], 999);
            if (total_both_sides >= gridDim.x - 1) {
                zeroed_scratch_buffer[0] = 0;
                zeroed_scratch_buffer[1] = 0;
                zeroed_scratch_buffer[2] = 0;
            }
            shared_sm_side_index = sm_side_index;
        }
        __syncthreads();
        sm_side_index = shared_sm_side_index;
        if (sm_side_index >= num_sm_per_side) {
            return;
        }
    } else {
        // Use static side index (doesn't work if some SMs are not avaialble or grimDim != SM count)
        sm_side_index = params.side_index[smid] >> 1;
        if (sm_side_index >= num_sm_per_side) {
            return;
        }
    }

    int group = threadIdx.y;
    int group_tid_offset = threadIdx.x;
    int num_groups_per_side = num_sm_per_side * blockDim.y;

    constexpr int CHUNK_BYTES = 4096;
    constexpr int CHUNK_VECTORS = CHUNK_BYTES / vector_bytes;
    constexpr int DOUBLE_CHUNK_VECTORS = 2 * CHUNK_VECTORS;
    constexpr size_type offset_per_group = (unrolled * DOUBLE_CHUNK_VECTORS);

    int num_double_chunks = num_vectors_16B / DOUBLE_CHUNK_VECTORS;
    int multi_chunks = num_double_chunks / unrolled;

    size_type offset_outer_loop = (offset_per_group * num_groups_per_side);
    offset_outer_loop = (reverse_order ? -offset_outer_loop : offset_outer_loop);

    int global_idx = (sm_side_index * blockDim.y) + group;
    int adjusted_global_idx = reverse_order ? (multi_chunks - global_idx - 1) : global_idx;
    size_type global_offset = adjusted_global_idx * offset_per_group + group_tid_offset;

    unsigned int base = (sideaware_for_o0 ? (unsigned long long)output0 : (unsigned long long)input0) & 0xFFFFFFFFULL;
    if constexpr (aligned_2MiB) {
        if (base & (2*1024*1024)) {
            sm_side ^= 1; // 2MiB aligned but not 4MiB aligned ==> invert side since bit 21 is in our custom hash
        }
        base = 0;
    } else {
        base /= vector_bytes;
    }

    size_type offsets[unrolled];
    vi0 in0[unrolled];
    vi1 in1[unrolled];
    vi2 in2[unrolled];
    vi3 in3[unrolled];
    vo0 out0;
    vo1 out1;
    vo2 out2;
    vo3 out3;

    #pragma unroll 1
    for (int i = global_idx; i < multi_chunks; i += num_groups_per_side, global_offset += offset_outer_loop) {
        #pragma unroll unrolled
        for (int j = 0; j < unrolled; j++) {
            size_type inner_offset = global_offset + (j * DOUBLE_CHUNK_VECTORS);
            unsigned int lsb_bits = base + (inner_offset & 0xFFFFFFFF);

            int side = __popc(lsb_bits & HASH) & 1;
            if constexpr (FORCE_WRONG_SIDE) side ^= 1;
            if constexpr (FORCE_RANDOM_SIDE) side = 0;

            int use_second_chunk = sm_side ^ side;
            size_type offset = inner_offset + (use_second_chunk * CHUNK_VECTORS);

            offsets[j] = offset;
            in0[j] = load<input_evict[0]>(input0 + offset);
            in1[j] = load<input_evict[1]>(input1 + offset);
            in2[j] = load<input_evict[2]>(input2 + offset);
            in3[j] = load<input_evict[3]>(input3 + offset);
        }

        #pragma unroll unrolled
        for (int j = 0; j < unrolled; j++) {
            size_type offset = offsets[j];
            vector_op(offset, vec_size, +out0, +out1, +out2, +out3, +in0[j], +in1[j], +in2[j], +in3[j], mem, val);
            store(output0 + offset, out0);
            store(output1 + offset, out1);
            store(output2 + offset, out2);
            store(output3 + offset, out3);
        }

        #pragma unroll unrolled
        for (int j = 0; j < unrolled; j++) {
            size_type offset = offsets[j];
            discard_inputs(input0 + offset, input1 + offset, input2 + offset, input3 + offset);
        }
    }

    if (group == 0) {
        int max_remaining_double_chunks = (num_double_chunks + 1) - (multi_chunks * unrolled);
        int start_sm_side_idx = num_sm_per_side - max_remaining_double_chunks;
        int idx = sm_side_index - start_sm_side_idx;
        if (idx >= 0) {
            size_type global_offset = (size_type)(idx + multi_chunks*unrolled) * DOUBLE_CHUNK_VECTORS;
            global_offset += group_tid_offset;

            unsigned int lsb_bits = base + (global_offset & 0xFFFFFFFF);
            int side = __popc(lsb_bits & HASH) & 1;

            int use_second_chunk = sm_side ^ side;
            size_type offset = global_offset + (use_second_chunk * CHUNK_VECTORS);
            if (offset < num_vectors_16B) {
                in0[0] = load<input_evict[0]>(input0 + offset);
                in1[0] = load<input_evict[1]>(input1 + offset);
                in2[0] = load<input_evict[2]>(input2 + offset);
                in3[0] = load<input_evict[3]>(input3 + offset);

                vector_op(offset, vec_size, +out0, +out1, +out2, +out3, +in0[0], +in1[0], +in2[0], +in3[0], mem, val);
                store(output0 + offset, out0);
                store(output1 + offset, out1);
                store(output2 + offset, out2);
                store(output3 + offset, out3);
                discard_inputs(input0 + offset, input1 + offset, input2 + offset, input3 + offset);
            }
        }
    }
}
)SIDEAWARE_KERNEL";

// ---------------------------------------------------------------------------
// Compile Time Settings
// ---------------------------------------------------------------------------
//#define DEBUG_PRINTF
constexpr bool   ALWAYS_OUTPUT_SASS = false;   // always output assembly to "sass" file (even if filename is empty)
constexpr bool   ALWAYS_TRY_CUDA_FREE = true;  // cudaFreeAsync for unknown pointers (e.g. if alloc threshold changes)
constexpr bool   ALWAYS_SET_DEVICE = true;     // ScopedSetDevice to set & restore device (is this really necesssary?)
constexpr int    DEFAULT_PARALLEL_CHUNKS = 4;  // number of 256 threads "groups" (blockDim.y) per SM (max 4 on H100)
constexpr bool   SYNC_ON_EVERY_ALLOC = false;  // sync when (pre)allocating memory to avoid confusing async errors
constexpr bool   ZERO_ON_PREALLOC = false;     // zero out memory when preallocating physical memory

constexpr int    MAX_SM = 640;                 // 640 SMs ought to be enough for anybody (kernels use real SM count)
constexpr int    FORCED_HASH = 0;              // 0xB3000 for H100, 0xAB000 for GH200 96GiB, 0x1EF000 for GB200
constexpr int    L2_SIDE_TEST_ITERATIONS = 25; // increases warmup time (at init & per page) but improves accuracy
constexpr int    SIDE_HASH_BIT = 21;           // we force bit 21 of the virtual address to track the side of the page
constexpr size_t CHUNK_BYTES = 4096;           // 4KiB (= granularity of side switch on H100/GB200)
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
// Global configuration variables
// ---------------------------------------------------------------------------
static bool g_use_compression = false; // Enable CUDA compressible memory (release unused memory to avoid mixed reuse)
static size_t g_custom_alloc_threshold = 8ULL * 1024ULL * 1024ULL; // use custom allocator above this
static size_t g_free_mapped_start_threshold = 16ULL * 1024ULL * 1024ULL * 1024ULL; // auto unmap on malloc above this
static size_t g_free_mapped_end_threshold   = 2ULL  * 1024ULL * 1024ULL * 1024ULL; // stop auto unmapping at this point
static int g_prealloc_extra_required = 1, g_prealloc_extra_alloc = 10; // configured by set_prealloc_config()
static std::string g_sass_filename; // If empty, don't output SASS. Set via set_output_sass().
static int g_num_devices = -1; // auto-init on 1st call to get_device_context()

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
    int side_summary[5];

    // GPU memory for side info
    unsigned int* gpu_allocator_metadata = nullptr;
    unsigned int* gpu_side_info = nullptr;
    unsigned char* gpu_side_index = nullptr;
    unsigned char* gpu_scratch_buffer = nullptr;
    unsigned char* gpu_scratch_zeroed_buffer = nullptr;
    // Unbacked virtual memory tracking
    unsigned char cpu_page_side[UNBACKED_VIRTUAL_PAGES];

    // Memory pools
    std::vector<CUmemGenericAllocationHandle> free_handles_side[2];
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
        CUmodule module[2] = {nullptr, nullptr};
        CUfunction funcs[4] = {nullptr,nullptr};
    };

    std::vector<KernelCacheEntry> kernel_cache;       // index = header ID (0 default)
    std::vector<std::string>      header_strings;     // same index – header source
    std::unordered_map<std::string,int> header_to_id; // reverse lookup
};

// ---------------------------------------------------------------------------
// Helper Functions & Classes
// ---------------------------------------------------------------------------
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
static void compileStringToCUBIN(CUdevice device, char **cubin_out, const char *source,
                              const char *header_code = nullptr, size_t *cubin_size_out = nullptr,
                              const char *include_path = "/usr/local/cuda/include/") {
    std::string source_code(source);
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
    checkCudaErrors(nvrtcCreateProgram(&program, final_source.c_str(), "side_aware", 0, nullptr, nullptr));
    nvrtcResult compile_res = nvrtcCompileProgram(program, opts_c.size(), opts_c.data());

    size_t log_size = 0;
    checkCudaErrors(nvrtcGetProgramLogSize(program, &log_size));
    if (log_size > 1) { // Print log even on success (for warnings)
        std::string log(log_size, '\0');
        checkCudaErrors(nvrtcGetProgramLog(program, &log[0]));
        std::cerr << "NVRTC Log (Side Aware):\n" << log << std::endl;
    }

    if (compile_res != NVRTC_SUCCESS) {
        const char* error_string = nvrtcGetErrorString(compile_res);
        std::cerr << "Error: NVRTC compilation failed for 'Side Aware"
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
        if (g_num_devices != 1 && ALWAYS_SET_DEVICE) {
            cudaGetDevice(&old_device);
            cudaSetDevice(new_device);
        }
    }
    ~ScopedSetDevice() {
        if (g_num_devices != 1 && ALWAYS_SET_DEVICE) {
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
        assert(offset * sizeof(int) < CHUNK_BYTES);

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
                if (!(toggle_bits & CHUNK_BYTES) || (toggle_bits & (CHUNK_BYTES - 1))) {
                    printf("\nERROR: CHUNK_BYTES %d doesn't work with hash %x\n\n", (int)CHUNK_BYTES, toggle_bits);
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

// NVRTC kernel compilation helper (for sideaware_elementwise)
static CUfunction sideaware_get_kernel(DeviceContext &ctx, int kernel_id, bool use_slow_path) {
    assert(kernel_id < ctx.kernel_cache.size() && kernel_id < ctx.header_strings.size());
    DeviceContext::KernelCacheEntry &entry = ctx.kernel_cache[kernel_id];

    int idx = use_slow_path ? 1 : 0; // select variant
    if (entry.funcs[idx] == nullptr) {
        assert(entry.module[idx] == nullptr);

        CUdevice cuDevice;
        cuCtxGetDevice(&cuDevice);

        const std::string& header_string = ctx.header_strings[kernel_id];
        assert(header_string.empty() == false);

        // Dynamically set MAX_SM and HASH for this device.
        unsigned int hash = ctx.cpu_side_info[OFFSET_SIDE_HASH_MASK] | (1u << SIDE_HASH_BIT);
        unsigned int hash_vector = hash / sizeof(uint4);
        std::string macro_header =
            "#define MAX_SM " + std::to_string(ctx.num_sms) + "\n" +
            "#define HASH " + std::to_string(hash_vector) + "\n";
        std::string variant_header = use_slow_path
            ? "using size_type = long long;\nconstexpr bool aligned_2MiB = false;\n"
            : "using size_type = int;\nconstexpr bool aligned_2MiB = true;\n";
        std::string combined_header = macro_header + variant_header + header_string;

        char *cubin = nullptr;
        compileStringToCUBIN(cuDevice, &cubin, SIDEAWARE_KERNEL_SOURCE, combined_header.c_str());
        entry.module[idx] = loadCUBIN(cubin, cuDevice);

        CUfunction ftmp;
        CUresult rc = cuModuleGetFunction(&ftmp, entry.module[idx], "sideaware_elementwise");
        if (rc != CUDA_SUCCESS) {
            const char *errStr = nullptr; cuGetErrorString(rc, &errStr);
            std::cerr << "Failed to get function: "
                      << (errStr ? errStr : "") << " (for kernel " << kernel_id
                      << ", variant " << (use_slow_path ? 1 : 0) << ")" << std::endl;
            std::abort();
        }
        entry.funcs[idx] = ftmp;
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

static CUmemAllocationProp get_allocation_constraints(DeviceContext& ctx, bool compression=false)
{
    compression &= ctx.compression_available;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ctx.device_id;
    prop.allocFlags.compressionType = compression ? CU_MEM_ALLOCATION_COMP_GENERIC : 0;
    return prop;
}

void DeviceContext::initialize(cudaStream_t stream) {
    if (initialized) return;
    ScopedSetDevice guard(device_id);

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

    init_allocation_constraints(*this);
    cudaMalloc(&gpu_allocator_metadata, PAGE_SIZE);
    cudaMemset(gpu_allocator_metadata, 0, PAGE_SIZE);
    assert((uintptr_t)gpu_allocator_metadata % PAGE_SIZE == 0);

    gpu_side_info = &gpu_allocator_metadata[16 * 1024];
    gpu_side_index = (unsigned char*)&gpu_allocator_metadata[32 * 1024];
    gpu_scratch_buffer = (unsigned char*)&gpu_allocator_metadata[64 * 1024];
    gpu_scratch_zeroed_buffer = (unsigned char*)&gpu_allocator_metadata[128 * 1024];

    init_side_info<<<num_sms, 512, 0, stream>>>(gpu_allocator_metadata, gpu_side_info, gpu_side_index);

    unsigned char cpu_side_index[MAX_SM];
    cudaMemcpyAsync(cpu_side_index, gpu_side_index, MAX_SM * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    cudaMemcpy(cpu_side_info, gpu_side_info, MAX_SM * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost); // syncs both

    for (int i = 0; i < MAX_SM; i++) {
        param_sm_side.side_index[i] = cpu_side_index[i];
    }
    header_strings.resize(1);
    kernel_cache.resize(1);
    header_strings[0] = SIDEAWARE_MEMCPY_HEADER; // ID 0 = memcpy

    initialized = true;
}

// ---------------------------------------------------------------------------
// Unmap free blocks if threshold is exceeded
// ---------------------------------------------------------------------------
static void unmap_free_allocations(DeviceContext& ctx, size_t start_threshold=0, size_t end_threshold=0) {
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
                ctx.free_handles_side[side].push_back(blk.handles[i]);
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
    unmap_free_allocations(ctx);

    size_t freed_memory = 0;
    for (int side = 0; side < 2; side++) {
        for (auto &h : ctx.free_handles_side[side]) {
            cuMemRelease(h);
            freed_memory += PAGE_SIZE;
        }
        ctx.free_handles_side[side].clear();
    }

    cudaDeviceSynchronize();
    return freed_memory;
}

// ---------------------------------------------------------------------------
// KEY FUNCTION: This is where the magic happens!
// Allocate physical memory and determine which side it's on and stores the
// handles in per-side "free pools" for future mapping to virtual memory
// (pre-allocate more than required because it causes a GPU sync)
// ---------------------------------------------------------------------------
static CUresult sideaware_preallocate(DeviceContext& ctx, int countNeeded, bool compression, cudaStream_t stream)
{
    CUmemAllocationProp prop = get_allocation_constraints(ctx, compression);
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

        // Zero out memory before using it
        if constexpr (ZERO_ON_PREALLOC) {
            lastError = cuMemsetD8(dptr, 0, batchAllocated * PAGE_SIZE);
            if (lastError != CUDA_SUCCESS) {
                printf("ERROR: Failed to zero out memory before test (this should never happen?!)\n");
                assert(false);
                return lastError;
            }
        }

        // Classify each page (side 0 or 1) by calling test_page_latency
        test_page_latency<<<1, 512, 0, stream>>>((unsigned int*)dptr, ctx.gpu_side_info,
                                                 ctx.gpu_scratch_buffer, batchAllocated);
        cudaMemcpy(ctx.cpu_page_side, ctx.gpu_scratch_buffer, batchAllocated, cudaMemcpyDeviceToHost);

        // Unmap + place each handle in the correct side's pool
        for (int i = 0; i < batchAllocated; i++) {
            cuMemUnmap(dptr + i * PAGE_SIZE, PAGE_SIZE);
            int side = ctx.cpu_page_side[i];
            ctx.free_handles_side[side].push_back(tempHandles[i]);
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

static CUresult sideaware_allocate_threshold(DeviceContext& ctx, size_t needed, bool compression, cudaStream_t stream)
{
    size_t needed_per_side = (needed + g_prealloc_extra_required) / 2;
    size_t free0 = ctx.free_handles_side[0].size();
    size_t free1 = ctx.free_handles_side[1].size();
    size_t needed_side0 = (needed_per_side > free0) ? (needed_per_side - free0) : 0;
    size_t needed_side1 = (needed_per_side > free1) ? (needed_per_side - free1) : 0;
    size_t needed_worst_case = max(needed_side0, needed_side1);
    size_t needed_both_sides = 2 * needed_worst_case;

    if (needed_both_sides > 0) {
        needed_both_sides += g_prealloc_extra_alloc;
        CUresult rc = sideaware_preallocate(ctx, needed_both_sides, compression, stream);
        return rc;
    }
    return CUDA_SUCCESS;
}

constexpr inline int side_from_virtual_address(uint64_t va)
{
    return (int)((va >> SIDE_HASH_BIT) & 1ULL);
}

static CUresult sideaware_allocate(DeviceContext& ctx, LargeAllocInfo &info, size_t size, bool compression)
{
    info.aligned_size = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    info.use_compression = (compression && ctx.compression_available);

    // Optional sync to make sure we catch all async errors before allocation
    if constexpr (SYNC_ON_EVERY_ALLOC) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            printf("ERROR: cudaDeviceSynchronize() failed before allocation, unrelated async error?\n");
            assert(false);
        }
    }

    // 1) Pre-check we have enough free handles, otherwise allocate more
    size_t num_pages = info.aligned_size / PAGE_SIZE;
    CUresult rc = sideaware_allocate_threshold(ctx, num_pages, info.use_compression, info.last_use_stream);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    // Determine the desired "start side" for this allocation size
    // If we haven't seen this size before, choose the side with fewer handles
    if (ctx.size_side_map.find(info.aligned_size) == ctx.size_side_map.end()) {
        int side0_free = ctx.free_handles_side[0].size();
        int side1_free = ctx.free_handles_side[1].size();
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
    int firstPageSide = side_from_virtual_address((uint64_t)basePtr);
    if (firstPageSide != desiredStartSide) {
        basePtr += PAGE_SIZE; // Skip to next 2MiB page
    }

    // Verify that this page has the desired side
    int basePageSide = side_from_virtual_address((uint64_t)basePtr);
    if (basePageSide != desiredStartSide) {
        // This should never happen with our hash function using bit 21?! (assuming SIDE_HASH_BIT = 21)
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
        int desiredSide = side_from_virtual_address(thisVa);

        // pop from the correct side if available
        std::vector<CUmemGenericAllocationHandle>* correctPool = &ctx.free_handles_side[desiredSide];
        std::vector<CUmemGenericAllocationHandle>* otherPool   = &ctx.free_handles_side[1 - desiredSide];

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
            ctx.free_handles_side[info.side_used[i]].push_back(info.handles[i]);
            // Cleanup on allocation failure
            for (size_t j = 0; j < i; j++) {
                cuMemUnmap(basePtr + j * PAGE_SIZE, PAGE_SIZE);
                ctx.free_handles_side[info.side_used[j]].push_back(info.handles[j]);
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
            ctx.free_handles_side[info.side_used[i]].push_back(info.handles[i]);
        }
        cuMemAddressFree(allocPtr, extraSize);
        return rc;
    }

    // 5) Optional sync to make sure we catch all async errors
    if constexpr (SYNC_ON_EVERY_ALLOC) {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            // Cleanup on allocation failure
            for (size_t i = 0; i < num_pages; i++) {
                cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
                ctx.free_handles_side[info.side_used[i]].push_back(info.handles[i]);
            }
            cuMemAddressFree(allocPtr, extraSize);
            return CUDA_ERROR_UNKNOWN;
        }
    }
    return CUDA_SUCCESS;
}

static void* sideaware_reuse_allocation(DeviceContext& ctx, size_t alignedSize, cudaStream_t currentStream)
{
    auto &vec = ctx.large_alloc_cache[alignedSize + (int)g_use_compression];
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

static void* sideaware_try_allocate(DeviceContext& ctx, size_t userSize, cudaStream_t stream)
{
    LargeAllocInfo info;
    info.user_requested = userSize;
    info.last_use_stream = stream;
    CUresult rc = sideaware_allocate(ctx, info, userSize, g_use_compression);
    if (rc != CUDA_SUCCESS)
        return nullptr;
    ctx.large_alloc_registry[info.base_ptr] = info;
    return info.base_ptr;
}

static void* sideaware_malloc_large(DeviceContext& ctx, size_t size, cudaStream_t stream)
{
    size_t alignedSize = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

    // 1) Try exact-size reuse
    void* p = sideaware_reuse_allocation(ctx, alignedSize, stream);
    if (p) return p;

    // 2) Unmap free pages over configurable threshold (then try allocating new pages for the 1st time)
    unmap_free_allocations(ctx, g_free_mapped_start_threshold, g_free_mapped_end_threshold);
    if (p = sideaware_try_allocate(ctx, size, stream)) return p;

    // 3) Unmap all free pages & try again
    unmap_free_allocations(ctx);
    if (p = sideaware_try_allocate(ctx, size, stream)) return p;

    // 4) Try fully releasing all unused memory & try again (this shouldn't be required?)
    release_unused_memory_device(ctx.device_id);
    return sideaware_try_allocate(ctx, size, stream);
}

static void sideaware_free_large(DeviceContext& ctx, void* ptr, cudaStream_t stream, bool try_cuda_free = false)
{
    if (!ptr) return;
    auto it = ctx.large_alloc_registry.find(ptr);
    if (it == ctx.large_alloc_registry.end()) {
        if (ALWAYS_TRY_CUDA_FREE || try_cuda_free) {
            cudaFreeAsync(ptr, stream);
        } else {
            printf("ERROR: Failed to find large allocation %p to free on device %d\n", ptr, ctx.device_id);
        }
        return;
    }

    LargeAllocInfo info = it->second;
    ctx.large_alloc_registry.erase(it);
    ctx.large_alloc_cache[info.aligned_size + (int)info.use_compression].push_back(info);
    ctx.total_mapped_free += info.aligned_size;
}

// ---------------------------------------------------------------------------
// PUBLIC API
// ---------------------------------------------------------------------------
extern "C" {

void* sideaware_malloc(size_t size, cudaStream_t stream) {
    debugf("sideaware_malloc(%zu, %p)\n", size, stream);
    return sideaware_malloc_large(get_device_context(), size, stream);
}

void sideaware_free(void* ptr, cudaStream_t stream) {
    debugf("sideaware_free(%p, %p)\n", ptr, stream);
    sideaware_free_large(get_device_context(), ptr, stream);
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
// PyTorch interface (torch.cuda.memory.CUDAPluggableAllocator)
// ---------------------------------------------------------------------------
void* sideaware_malloc_auto(size_t size, int device, cudaStream_t stream) {
    debugf("sideaware_malloc_auto(%zu, %d, %p)\n", size, device, stream);
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

        unmap_free_allocations(ctx);
        err = cudaMallocAsync(&p, size, stream);
        if (err != cudaSuccess) {
            p = nullptr;
        }
    }
    return p;
}

void sideaware_free_auto(void* ptr, size_t size, int device, cudaStream_t stream) {
    debugf("sideaware_free_auto(%p, %zu, %d, %p)\n", ptr, size, device, stream);

    if (!ptr) return;
    DeviceContext& ctx = get_device_context(device);
    ScopedSetDevice guard(device);

    bool try_cuda_free = size >= g_custom_alloc_threshold;
    sideaware_free_large(ctx, ptr, stream, try_cuda_free);
}

// ---------------------------------------------------------------------------
// Side Aware Elementwise GPU kernel launchers
// kernel_id selects runtime‑compiled module to use (0 = default memcpy)
// ---------------------------------------------------------------------------
void sideaware_elementwise(int kernel_id,
                           size_t num_bytes,
                           void* out0, void* out1, void* out2, void* out3,
                           const void* in0, const void* in1, const void* in2, const void* in3,
                           void* sideband_memory, int sideband_value,
                           int parallel_chunks, int forced_per_side,
                           int device, cudaStream_t stream)
{
    if (num_bytes == 0 || out0 == nullptr || in0 == nullptr) return;
    ScopedSetDevice guard(device);
    DeviceContext& ctx = get_device_context(device);
    CUstream cuStream = reinterpret_cast<CUstream>(stream);

    assert(kernel_id >= 0 && kernel_id < ctx.kernel_cache.size());
    assert(forced_per_side < ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE]);
    assert((intptr_t)out0 % 16 == 0);
    assert((intptr_t)in0 % 16 == 0);
    assert(num_bytes % 16 == 0);

    unsigned int sm_per_side = (forced_per_side > 0) ? forced_per_side : ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
    int num_groups = parallel_chunks > 0 ? parallel_chunks : DEFAULT_PARALLEL_CHUNKS;
    long long num_vectors_64b = num_bytes / sizeof(uint4);
    int num_vectors_32b = (int)num_vectors_64b;

    bool misaligned_2MiB = ((intptr_t)in0 % (2ULL*1024*1024)) || ((intptr_t)out0 % (2ULL*1024*1024));
    bool need_64b_indexing = (num_bytes < 2ULL*1024*1024*1024);
    bool use_slow_path = need_64b_indexing || misaligned_2MiB;

    void* args[14];
    args[0] = use_slow_path ? (void*)&num_vectors_64b : (void*)&num_vectors_32b;
    args[1] = &out0;
    args[2] = &out1;
    args[3] = &out2;
    args[4] = &out3;
    args[5] = &in0;
    args[6] = &in1;
    args[7] = &in2;
    args[8] = &in3;
    args[9] = &sideband_memory;
    args[10] = &sideband_value;
    args[11] = &sm_per_side;
    args[12] = &ctx.gpu_scratch_zeroed_buffer;
    args[13] = &ctx.param_sm_side;

    CUfunction kernel = sideaware_get_kernel(ctx, kernel_id, use_slow_path);
    cuLaunchKernel(kernel, ctx.num_sms, 1, 1, 256, num_groups, 1, 0, cuStream, args, nullptr);
}

void sideaware_one_to_one(int kernel_id, size_t num_bytes, void* out0, const void* in0, int device, cudaStream_t stream) {
    sideaware_elementwise(kernel_id, num_bytes, out0, 0, 0, 0, in0, 0, 0, 0, 0, 0, 0, 0, device, stream);
}

void sideaware_memcpy(void* out, const void* in, size_t num_bytes, int device, cudaStream_t stream) {
    sideaware_elementwise(0, num_bytes, out, 0, 0, 0, in, 0, 0, 0, 0, 0, 0, 0, device, stream);
}

// ---------------------------------------------------------------------------
// Compile custom elementwise kernel with provided header (compiled with NVRTC)
// ---------------------------------------------------------------------------
int sideaware_compile(const char* header, bool precompile) {
    if (!header || strlen(header)==0) {
        return 0; // ID 0 = memcpy
    }

    int kernel_idx = -1;
    std::string hdr_str(header);

    for (size_t i = 0; i < g_deviceContexts.size(); i++) {
        DeviceContext& ctx = g_deviceContexts[i];
        auto it = ctx.header_to_id.find(hdr_str);
        if (it != ctx.header_to_id.end()) {
            assert(i == 0 || kernel_idx == it->second);
            kernel_idx = it->second;
            continue;
        }

        int new_id = ctx.header_strings.size();
        ctx.header_strings.push_back(hdr_str);
        ctx.kernel_cache.emplace_back();
        ctx.header_to_id[hdr_str] = new_id;

        assert(i == 0 || kernel_idx == new_id);
        kernel_idx = new_id;
    }

    // Optionally precompile the kernel on the current device
    if (precompile) {
        DeviceContext& ctx = get_device_context();
        assert(sideaware_get_kernel(ctx, kernel_idx, false) != nullptr);
        assert(sideaware_get_kernel(ctx, kernel_idx, true) != nullptr);
    }
    return kernel_idx;
}

// ---------------------------------------------------------------------------
// Configuration functions
// ---------------------------------------------------------------------------
void use_compression(bool value) {
    g_use_compression = value;
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

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------
void fill_sm_sides_tensor(unsigned char* gpu_tensor) {
    DeviceContext& ctx = get_device_context();
    cudaError_t err = cudaMemcpy(gpu_tensor, ctx.gpu_side_index, ctx.num_sms, cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);
}

const int* get_sm_side_summary()
{
    DeviceContext& ctx = get_device_context();
    ctx.side_summary[0] = ctx.num_sms;
    ctx.side_summary[1] = ctx.cpu_side_info[OFFSET_NUM_SM_SIDE0];
    ctx.side_summary[2] = ctx.cpu_side_info[OFFSET_NUM_SM_SIDE1];
    ctx.side_summary[3] = ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
    ctx.side_summary[4] = ctx.cpu_side_info[OFFSET_SIDE_HASH_MASK];
    return ctx.side_summary;
}

int get_num_sms() { return get_device_context().num_sms; }
int get_num_sm_side0() { return get_device_context().cpu_side_info[OFFSET_NUM_SM_SIDE0]; }
int get_num_sm_side1() { return get_device_context().cpu_side_info[OFFSET_NUM_SM_SIDE1]; }
int get_min_sm_per_side() { return get_device_context().cpu_side_info[OFFSET_MIN_SM_PER_SIDE]; }
int get_hash_mask() { return get_device_context().cpu_side_info[OFFSET_SIDE_HASH_MASK]; }

} // extern "C"