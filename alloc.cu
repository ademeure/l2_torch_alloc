// Example command:
// nvcc --gpu-architecture=native -Xcompiler -fPIC -shared alloc.cu -o alloc.so -lcuda && python test.py

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda.h>

// ---------------------------------------------------------------------------
// Compile Time Settings
// ---------------------------------------------------------------------------
constexpr size_t CUSTOM_ALLOC_THRESHOLD = 4ULL * 1024ULL * 1024ULL; // use custom allocator above this
constexpr size_t FREE_MAPPED_START_THRESHOLD = 8192ULL * 1024ULL * 1024ULL; // auto unmap on malloc above this
constexpr size_t FREE_MAPPED_END_THRESHOLD   = 2048ULL * 1024ULL * 1024ULL; // stop auto unmapping at this point

constexpr size_t CHUNK_SIZE = 4096;            // 4KiB (= granularity of side switch on H100/GB200)
constexpr size_t PAGE_SIZE  = 2 * 1024 * 1024; // 2MiB (= granularity of NVIDIA MMU pages)
constexpr int MAX_SM = 200; // enough for all NVIDIA GPUs up to GB300 (but not for e.g. MI300X!)

constexpr int L2_SIDE_TEST_ITERATIONS = 25; // increases warmup time (at init & per page) but improves accuracy
constexpr size_t UNBACKED_VIRTUAL_PTR_SIZE = 2048ULL * 1024ULL * 1024ULL; // memory allocated in one go
constexpr size_t UNBACKED_VIRTUAL_PAGES = UNBACKED_VIRTUAL_PTR_SIZE / PAGE_SIZE;

// For debugging & performance analysis only
constexpr bool DETERMINE_HASH_DYNAMICALLY = true;
constexpr int  DEFAULT_HASH = 0x2B3000; // H100
constexpr bool FORCE_RANDOM_SIDE = false;
constexpr bool FORCE_WRONG_SIDE = false;
//#define DEBUG_PRINTF

// ---------------------------------------------------------------------------
// Helpers & Static Variables
// ---------------------------------------------------------------------------
static int g_num_devices = -1; // initialised on 1st call to getDeviceContext()
static int g_preAllocX = 1, g_preAllocY = 10; // configured by set_prealloc_config() (TODO: rename & explain)

template<class T>
constexpr inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

template<class... Ts>
__device__ __host__ static void debugf(const char* fmt, Ts... args) {
#ifdef DEBUG_PRINTF
    std::printf(fmt, args...);
#endif
}

class ScopedSetDevice {
public:
    explicit ScopedSetDevice(int new_device) {
        if (g_num_devices != 1) {
            cudaGetDevice(&old_);
        }
        cudaSetDevice(new_device);
    }
    ~ScopedSetDevice() {
        if (g_num_devices != 1) {
            cudaSetDevice(old_);
        }
    }
private:
    int old_;
};

// Offsets into the cpu_side_info array
constexpr int OFFSET_ZEROED_COUNTER    = MAX_SM;
constexpr int OFFSET_AVERAGE_LATENCY   = MAX_SM + 1;
constexpr int OFFSET_SIDE_HASH_MASK    = MAX_SM + 2;
constexpr int OFFSET_NUM_SM_SIDE0      = MAX_SM + 3;
constexpr int OFFSET_NUM_SM_SIDE1      = MAX_SM + 4;
constexpr int OFFSET_MIN_SM_PER_SIDE   = MAX_SM + 5;

// Passed as a single argument to side_aware_memcpy
typedef struct {
    unsigned char side_index[MAX_SM];
} param_sm_side_t;

// ---------------------------------------------------------------------------
// A single struct that holds everything needed for chunk-based allocation
// ---------------------------------------------------------------------------
struct LargeAllocInfo {
    size_t userRequested   = 0;
    void*  basePtr         = nullptr;  // Points to the correctly aligned section for mapping
    void*  allocPtr        = nullptr;  // Original pointer from cuMemAddressReserve
    size_t alignedSize     = 0;
    bool   useCompression  = false;
    cudaStream_t lastUseStream = 0;
    // One handle per 2MiB chunk:
    std::vector<CUmemGenericAllocationHandle> handles;

    // NEW: Track which side was actually used for each chunk so we know where to return it:
    std::vector<int> sideUsed; // 0 or 1 for each chunk
};
// ---------------------------------------------------------------------------
// Device Context structure to track per-device data
// ---------------------------------------------------------------------------
struct DeviceContext {
    // Device properties
    int device_id = -1;
    int num_sms = 0;
    bool initialized = false;

    // Side info
    int cpu_side_info[MAX_SM * 2];
    param_sm_side_t param_sm_side;

    // GPU memory for side info
    unsigned int* gpu_allocator_metadata = nullptr;
    unsigned int* gpu_side_info = nullptr;
    unsigned char* gpu_side_index = nullptr;
    unsigned char* gpu_scratch_buffer = nullptr;

    // Unbacked virtual memory tracking
    unsigned char cpu_page_side[UNBACKED_VIRTUAL_PAGES];

    // Memory pools
    std::vector<CUmemGenericAllocationHandle> freeHandlesSide0;
    std::vector<CUmemGenericAllocationHandle> freeHandlesSide1;
    std::unordered_map<size_t, std::vector<LargeAllocInfo>> largeAllocCache;
    std::unordered_map<void*, LargeAllocInfo> largeAllocRegistry;

    // State tracking
    size_t totalMappedFree = 0;
    bool crossSideWarningIssued = false;
    bool compressionAvailable = false;
    std::unordered_map<size_t, int> sizeSideMap;

    // Initialize the context for a specific device
    void initialize(cudaStream_t stream);
};

// ---------------------------------------------------------------------------
// Key function defined early as this template function cannot be defined in the extern "C" block
// The wrapper function calling this is defined at the very end (needs access to other variables)
// ---------------------------------------------------------------------------

template<bool input_aligned_2mib=false, int FORCED_UNROLL=4, typename size_type=size_t>
__global__ __launch_bounds__(1024, 1, 1) void side_aware_memcpy(uint4 * __restrict__ output,
                                                                const uint4 * __restrict__ input,
                                                                size_type num_bytes, int hash, int num_sm_per_side,
                                                                __grid_constant__ const param_sm_side_t params) {
    int smid;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

    // Get this SM's side (0 or 1) and its index within that side
    int sm_side = params.side_index[smid] & 1;
    int sm_side_index = params.side_index[smid] >> 1;

    // Exit early if there's an unbalanced number of SMs on each side, and this is an "extra SM" on the larger side
    if (sm_side_index >= num_sm_per_side) {
        __nanosleep(1000);
        return;
    }

    // TODO: could we calculate all of this stuff on the CPU and pass it as a parameter?
    // Each "group" is 256 threads processing 16 bytes each = 4096 bytes
    unsigned int num_groups = blockDim.x / 256;
    unsigned int group = threadIdx.x / 256;
    unsigned int group_tid = threadIdx.x % 256;
    unsigned int group_tid_offset = group_tid * 16;
    unsigned int num_groups_per_side = num_sm_per_side * num_groups;
    unsigned int global_idx = (sm_side_index * num_groups) + group;

    // We support an arbitrary number of bytes with partial chunks and out-of-bounds checking
    unsigned int num_double_chunks = num_bytes / (2 * CHUNK_SIZE);
    unsigned int multi_chunks = num_double_chunks / FORCED_UNROLL;
    unsigned int input_base = reinterpret_cast<intptr_t>(input) & 0xFFFFFFFF;

    size_type offset_per_j = 2 * CHUNK_SIZE;
    size_type offset_per_i = (FORCED_UNROLL * CHUNK_SIZE * 2);
    size_type offset_per_i_iter = (offset_per_i * num_groups_per_side) - offset_per_i;
    size_type byte_offset = global_idx * offset_per_i + group_tid_offset;

    for (unsigned int i = global_idx; i < multi_chunks; i += num_groups_per_side, byte_offset += offset_per_i_iter) {
        size_type offsets[FORCED_UNROLL];
        uint4 inputs[FORCED_UNROLL];
        #pragma unroll FORCED_UNROLL
        for (int j = 0; j < FORCED_UNROLL; j++, byte_offset += offset_per_j) {
            // Determine the side of the 1st 4KiB chunk in the 8KiB "double chunk"
            int lsb_side_bits;
            if constexpr (input_aligned_2mib) { lsb_side_bits = byte_offset & hash; }
            else { lsb_side_bits = (input_base + (byte_offset & 0xFFFFFFFF)) & hash; }

            int side = __popc(lsb_side_bits) & 1;
            if constexpr (FORCE_RANDOM_SIDE) {
                side = 0;
            } else if constexpr (FORCE_WRONG_SIDE) {
                side ^= 1;
            }

            // Switch to the 2nd 4KiB chunk in a 8KiB "double chunk" if the 1st 4KiB is on the other side
            // The paired SM from the other side will be responsible for the opposite 4KiB chunk
            // (since these two 4KiB chunks are always from opposite sides)
            // Optimized version of: "if (side != sm_side) byte_offset += CHUNK_SIZE;"
            int add_chunk_size = sm_side ^ side;
            size_type offset = byte_offset + (add_chunk_size * CHUNK_SIZE);

            offset /= sizeof(uint4);
            offsets[j] = offset;
            inputs[j] = input[offset];
        }
        #pragma unroll FORCED_UNROLL
        for (int j = 0; j < FORCED_UNROLL; j++) {
            output[offsets[j]] = inputs[j];
        }
    }

    // Process the remaining data that isn't a multiple of (2 * CHUNK_SIZE * FORCED_UNROLL)
    // 8KiB per threadgroup in the last threadgroups/SMs least likely to need to run the last wave
    if (group == 0) {
        int max_remaining_double_chunks = (num_double_chunks + 1) - (multi_chunks * FORCED_UNROLL);
        int start_sm_side_idx = num_sm_per_side - max_remaining_double_chunks;
        int idx = sm_side_index - start_sm_side_idx;
        if (idx >= 0) {
            size_type byte_offset = (idx + multi_chunks * FORCED_UNROLL) * (2 * CHUNK_SIZE) + group_tid * 16;
            int lsb_side_bits = (((input_aligned_2mib ? 0 : input_base) + (byte_offset & 0xFFFFFFFF)) & hash);
            int side = (__popc(lsb_side_bits) & 1);
            if (side != sm_side) {
                byte_offset += CHUNK_SIZE;
            }
            if (byte_offset + sizeof(uint4) <= num_bytes) {
                output[byte_offset / sizeof(uint4)] = input[byte_offset / sizeof(uint4)];
            }
        } else if (idx == -1 && sm_side == 0) {
            // Process the final partial uint4 (when the number of bytes is not a multiple of 16)
            size_type byte_offset = threadIdx.x + num_bytes - (num_bytes % sizeof(uint4));
            if(byte_offset < num_bytes) {
                ((unsigned char*)output)[byte_offset] = ((unsigned char*)input)[byte_offset];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Device function & kernel for side info
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

            // SM0 is always side 0
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

            if constexpr (DETERMINE_HASH_DYNAMICALLY) {
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
                side_info[OFFSET_SIDE_HASH_MASK] = DEFAULT_HASH;
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
                   smid, i,
                   (float)total_latency / (float)L2_SIDE_TEST_ITERATIONS,
                   total_latency, page_side[i]);
        }
    }
}

__global__ void test_single_address_latency(unsigned int* ptr, unsigned int *side_info)
{
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        if (smid != 0) { return; }

        int total_latency = test_latency_l2(ptr, 0);
        int average_latency = side_info[OFFSET_AVERAGE_LATENCY];
        intptr_t ptr_int = reinterpret_cast<intptr_t>(ptr);
        int expected_side = ((ptr_int >> 21) & 1) ^ ((ptr_int >> 24) & 1);

        printf("total_latency: %d, average_latency: %d ==> %d (expected: %d ===> AGREEMENT: %s)\n",
               total_latency, average_latency, (total_latency > average_latency), expected_side,
               (total_latency > average_latency) == expected_side ? "YES" : "NO");
    } else if (threadIdx.x >= 32) {
        __nanosleep(10000);
    }
}

extern "C" {

// ---------------------------------------------------------------------------
// Global variables
// ---------------------------------------------------------------------------
// Device contexts array - indexed by device ID
static std::vector<DeviceContext> g_deviceContexts;

// Get the device context for a given device ID (and initialize if never used before)
static DeviceContext& getDeviceContext(int device=-1) {
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

// ---------------------------------------------------------------------------
// Query the device allocation constraints
// ---------------------------------------------------------------------------
static void init_allocation_constraints(DeviceContext& ctx)
{
    int comp_available;
    cuDeviceGetAttribute(&comp_available, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, ctx.device_id);
    ctx.compressionAvailable = (comp_available != 0);

    // Verify granularity equals PAGE_SIZE
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ctx.device_id;

    size_t granularity;
    CUresult res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(res == CUDA_SUCCESS && granularity == PAGE_SIZE);

    if (ctx.compressionAvailable) {
        prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        assert(res == CUDA_SUCCESS && granularity == PAGE_SIZE);
    }
}

static CUmemAllocationProp get_allocation_constraints(DeviceContext& ctx, bool use_compression=false)
{
    use_compression = use_compression && ctx.compressionAvailable;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ctx.device_id;
    prop.allocFlags.compressionType = use_compression ? CU_MEM_ALLOCATION_COMP_GENERIC : 0;

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

    initialized = true;
}

// ---------------------------------------------------------------------------
// Unmap free blocks if threshold is exceeded
// ---------------------------------------------------------------------------
static void unmapFreeLargeAllocations(DeviceContext& ctx, size_t start_threshold=0, size_t end_threshold=0) {
    if (ctx.totalMappedFree <= start_threshold) return;

    ScopedSetDevice guard(ctx.device_id);
    cudaDeviceSynchronize();

    for (auto &kv : ctx.largeAllocCache) {
        auto &vec = kv.second;
        auto it = vec.begin();

        while (it != vec.end() && ctx.totalMappedFree > end_threshold) {
            LargeAllocInfo &blk = *it;
            size_t size = blk.alignedSize;
            size_t num_pages = blk.alignedSize / PAGE_SIZE;
            CUdeviceptr base = (CUdeviceptr)blk.basePtr;

            for (size_t i = 0; i < num_pages; i++) {
                cuMemUnmap(base + i * PAGE_SIZE, PAGE_SIZE);
                // Return handle to the side it was allocated from:
                int side = blk.sideUsed[i];
                if (side == 0) {
                    ctx.freeHandlesSide0.push_back(blk.handles[i]);
                } else {
                    ctx.freeHandlesSide1.push_back(blk.handles[i]);
                }
            }
            // Use allocPtr for freeing address space, not basePtr
            cuMemAddressFree((CUdeviceptr)blk.allocPtr, blk.alignedSize + 2 * PAGE_SIZE);

            assert(ctx.totalMappedFree >= size);
            ctx.totalMappedFree -= size;
            it = vec.erase(it);
        }
        if (ctx.totalMappedFree <= end_threshold)
            break;
    }
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Pre-allocate new physical handles
// ---------------------------------------------------------------------------
static CUresult preAllocateHandles(DeviceContext& ctx, int countNeeded, bool useCompression, cudaStream_t stream)
{
    if (countNeeded == 0) return CUDA_SUCCESS;

    ScopedSetDevice guard(ctx.device_id);

    CUmemAllocationProp prop = get_allocation_constraints(ctx, useCompression);
    CUresult lastError = CUDA_SUCCESS;
    int totalAllocated = 0;

    // We'll do it in batches so we don't exceed UNBACKED_VIRTUAL_PAGES at once.
    while (countNeeded > 0) {
        int batch = std::min<int>(countNeeded, UNBACKED_VIRTUAL_PAGES);
        int batchAllocated = 0;

        // Reserve a temporary VA space for `batch` pages:
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

        // If we allocated nothing in this batch, clean up and continue
        if (batchAllocated == 0) {
            cuMemAddressFree(dptr, batch * PAGE_SIZE);
            break;
        }

        // Set access for everything we allocated
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

        // Zero out memory before test (but continue even if this fails)
        cudaDeviceSynchronize();
        lastError = cuMemsetD8(dptr, 0, batchAllocated * PAGE_SIZE);
        if (lastError != CUDA_SUCCESS) {
            printf("ERROR: Failed to zero out memory before test (this should never happen?!)\n");
            assert(false);
            return lastError;
        }

        // Now classify each page by calling test_page_latency:
        test_page_latency<<<1, 512, 0, stream>>>(
            reinterpret_cast<unsigned int*>(dptr), ctx.gpu_side_info, ctx.gpu_scratch_buffer, batchAllocated);
        cudaMemcpy(ctx.cpu_page_side, ctx.gpu_scratch_buffer, batchAllocated, cudaMemcpyDeviceToHost);

        // Unmap + place each handle in correct side's pool
        for (int i = 0; i < batchAllocated; i++) {
            cuMemUnmap(dptr + i * PAGE_SIZE, PAGE_SIZE);
            int side = ctx.cpu_page_side[i];
            if (side == 0) {
                ctx.freeHandlesSide0.push_back(tempHandles[i]);
            } else {
                ctx.freeHandlesSide1.push_back(tempHandles[i]);
            }
        }
        cuMemAddressFree(dptr, batch * PAGE_SIZE);

        totalAllocated += batchAllocated;
        countNeeded -= batchAllocated;

        if (batchAllocated < batch) {
            break; // probably out of resources
        }
    }

    // If we allocated anything at all, consider it at least a partial success
    if (totalAllocated > 0) {
        return CUDA_SUCCESS;
    }
    return lastError;
}

// ---------------------------------------------------------------------------
// Ensure we have enough free handles
// ---------------------------------------------------------------------------
static CUresult ensureFreeHandlesAvailable(DeviceContext& ctx, size_t needed, bool useCompression, cudaStream_t stream)
{
    size_t needed_per_side = (needed + g_preAllocX) / 2;
    size_t free0 = ctx.freeHandlesSide0.size();
    size_t free1 = ctx.freeHandlesSide1.size();
    size_t needed_side0 = (needed_per_side > free0) ? (needed_per_side - free0) : 0;
    size_t needed_side1 = (needed_per_side > free1) ? (needed_per_side - free1) : 0;
    size_t needed_worst_case = max(needed_side0, needed_side1);
    size_t needed_both_sides = 2 * needed_worst_case;

    if (needed_both_sides > 0) {
        needed_both_sides += (g_preAllocY - g_preAllocX);
        CUresult rc = preAllocateHandles(ctx, needed_both_sides, useCompression, stream);
        return rc;
    }
    return CUDA_SUCCESS;
}

constexpr inline int pickSideFromVA(uint64_t va)
{
    // TODO: Make this work with a more complex hash, had issues with bit21+24
    // if ((va >> 21) & 1) == ((va >> 24) & 1) => side0 else side1
    int bit21 = (int)((va >> 21) & 1ULL);
    //int bit24 = (int)((va >> 24) & 1ULL);
    //return (bit21 == bit24) ? 0 : 1;
    return bit21;
}

// ---------------------------------------------------------------------------
// The allocateCompressible function
// ---------------------------------------------------------------------------
static CUresult allocateCompressible(DeviceContext& ctx, LargeAllocInfo &info, size_t size, bool use_compression=false)
{
    ScopedSetDevice guard(ctx.device_id);

    info.alignedSize = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    info.useCompression = (use_compression && ctx.compressionAvailable);

    // 1) Pre-check we have enough free handles, otherwise allocate more:
    size_t num_pages = info.alignedSize / PAGE_SIZE;
    CUresult rc = ensureFreeHandlesAvailable(ctx, num_pages, info.useCompression, info.lastUseStream);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    // Determine the desired "start side" for this allocation size
    // If we haven't seen this size before, choose the side with fewer handles
    if (ctx.sizeSideMap.find(info.alignedSize) == ctx.sizeSideMap.end()) {
        int side0_free = ctx.freeHandlesSide0.size();
        int side1_free = ctx.freeHandlesSide1.size();
        ctx.sizeSideMap[info.alignedSize] = (side0_free >= side1_free) ? 0 : 1;
    }
    int desiredStartSide = ctx.sizeSideMap[info.alignedSize];

    // 2) Reserve VA space with EXTRA space (2 extra pages = 4MiB)
    // This ensures we'll find a properly-aligned section with the desired side
    CUdeviceptr allocPtr = 0;
    size_t extraSize = info.alignedSize + 2 * PAGE_SIZE; // Add 4MiB extra space
    rc = cuMemAddressReserve(&allocPtr, extraSize, 0, 0, 0);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    info.allocPtr = reinterpret_cast<void*>(allocPtr);
    CUdeviceptr basePtr = allocPtr;

    // If the first page is not on the desired side, try the second page
    int firstPageSide = pickSideFromVA((uint64_t)basePtr);
    if (firstPageSide != desiredStartSide) {
        basePtr += PAGE_SIZE; // Skip to next 2MiB page
    }

    // Verify that this page or the 3rd page has the desired side
    int basePageSide = pickSideFromVA((uint64_t)basePtr);
    if (basePageSide != desiredStartSide) {
        basePtr += PAGE_SIZE;
        int thirdPageSide = pickSideFromVA((uint64_t)basePtr);
        if (thirdPageSide != desiredStartSide) {
            // This should never happen with our 1-bit or 2-bit hash functions?!
            printf("ERROR: Failed to find desired side in 3 consecutive 2MiB pages\n");
            cuMemAddressFree(allocPtr, extraSize);
            return CUDA_ERROR_UNKNOWN;
        }
    }

    // Set the basePtr to point to the section with the desired side
    info.basePtr = reinterpret_cast<void*>(basePtr);
    info.handles.resize(num_pages);
    info.sideUsed.resize(num_pages);

    // 3) Map each chunk from whichever side is indicated by bits(21,24)
    for (size_t i = 0; i < num_pages; i++) {
        uint64_t thisVa = (uint64_t)(basePtr + i * PAGE_SIZE);
        int desiredSide = pickSideFromVA(thisVa);

        // pop from the correct side if available
        std::vector<CUmemGenericAllocationHandle>* correctPool =
            (desiredSide == 0) ? &ctx.freeHandlesSide0 : &ctx.freeHandlesSide1;
        std::vector<CUmemGenericAllocationHandle>* otherPool   =
            (desiredSide == 0) ? &ctx.freeHandlesSide1 : &ctx.freeHandlesSide0;

        if (!correctPool->empty()) {
            info.handles[i] = correctPool->back();
            correctPool->pop_back();
            info.sideUsed[i] = desiredSide;
        } else {
            // Try to fallback to the other side
            if (otherPool->empty()) {
                // We've run out of GPU memory on both sides
                cuMemAddressFree(allocPtr, extraSize);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            // Warn the first time this happens
            if (!ctx.crossSideWarningIssued) {
                printf("WARNING: Cross-side handle usage on device %d!\n", ctx.device_id);
                ctx.crossSideWarningIssued = true;
            }
            info.handles[i] = otherPool->back();
            otherPool->pop_back();
            info.sideUsed[i] = desiredSide;
        }

        rc = cuMemMap(basePtr + i * PAGE_SIZE, PAGE_SIZE, 0, info.handles[i], 0);
        if (rc != CUDA_SUCCESS) {
            // Return the handle to whichever side we took it from
            if (info.sideUsed[i] == 0) {
                ctx.freeHandlesSide0.push_back(info.handles[i]);
            } else {
                ctx.freeHandlesSide1.push_back(info.handles[i]);
            }
            // Cleanup on allocation failure
            for (size_t j = 0; j < i; j++) {
                cuMemUnmap(basePtr + j * PAGE_SIZE, PAGE_SIZE);
                if (info.sideUsed[j] == 0) {
                    ctx.freeHandlesSide0.push_back(info.handles[j]);
                } else {
                    ctx.freeHandlesSide1.push_back(info.handles[j]);
                }
            }
            cuMemAddressFree(allocPtr, extraSize);
            return rc;
        }
    }

    // 4) Set access
    CUmemAccessDesc accessDesc;
    accessDesc.location.id   = ctx.device_id;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    rc = cuMemSetAccess(basePtr, info.alignedSize, &accessDesc, 1);
    if (rc != CUDA_SUCCESS) {
        // Cleanup on allocation failure
        for (size_t i = 0; i < num_pages; i++) {
            cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
            if (info.sideUsed[i] == 0) ctx.freeHandlesSide0.push_back(info.handles[i]);
            else                      ctx.freeHandlesSide1.push_back(info.handles[i]);
        }
        cuMemAddressFree(allocPtr, extraSize);
        return rc;
    }

    // 5) Zero memory to be safe (TODO: make this configurable)
    cudaDeviceSynchronize();
    rc = cuMemsetD8(basePtr, 0, info.alignedSize);
    if (rc != CUDA_SUCCESS) {
        // Cleanup on allocation failure
        for (size_t i = 0; i < num_pages; i++) {
            cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
            if (info.sideUsed[i] == 0) ctx.freeHandlesSide0.push_back(info.handles[i]);
            else                      ctx.freeHandlesSide1.push_back(info.handles[i]);
        }
        cuMemAddressFree(allocPtr, extraSize);
        return rc;
    }

    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// LargeAlloc: Try to reuse from cache or allocate new
// ---------------------------------------------------------------------------
static void* tryReuseLargeAlloc(DeviceContext& ctx, size_t alignedSize, cudaStream_t currentStream)
{
    auto &vec = ctx.largeAllocCache[alignedSize];
    if (vec.empty()) return nullptr;

    LargeAllocInfo info = vec.back();
    assert(ctx.totalMappedFree >= info.alignedSize);
    ctx.totalMappedFree -= info.alignedSize;
    vec.pop_back();

    // If different stream, we might need to synchronize
    // TODO: is this safe or do we need a full device synchronization?
    if (info.lastUseStream != currentStream) {
        cudaStreamSynchronize(info.lastUseStream);
        info.lastUseStream = currentStream;
    }

    // Insert back into active registry
    ctx.largeAllocRegistry[info.basePtr] = info;
    return info.basePtr;
}

static size_t release_unused_memory_device(int device) {
    DeviceContext& ctx = getDeviceContext(device);
    ScopedSetDevice guard(device);

    // Unmap all cached blocks then release handles from side0/side1 pools
    unmapFreeLargeAllocations(ctx);

    size_t freed_memory = 0;
    for (auto &h : ctx.freeHandlesSide0) {
        cuMemRelease(h);
        freed_memory += PAGE_SIZE;
    }
    ctx.freeHandlesSide0.clear();

    for (auto &h : ctx.freeHandlesSide1) {
        cuMemRelease(h);
        freed_memory += PAGE_SIZE;
    }
    ctx.freeHandlesSide1.clear();

    cudaDeviceSynchronize();
    return freed_memory;
}


static void* allocateNewLargeBlock(DeviceContext& ctx, size_t userSize, cudaStream_t stream)
{
    LargeAllocInfo info;
    info.userRequested = userSize;
    info.lastUseStream = stream;
    CUresult rc = allocateCompressible(ctx, info, userSize);
    if (rc != CUDA_SUCCESS)
        return nullptr;
    ctx.largeAllocRegistry[info.basePtr] = info;
    return info.basePtr;
}

static void* myMallocLarge(DeviceContext& ctx, size_t size, cudaStream_t stream)
{
    size_t alignedSize = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

    // 1) Try exact-size reuse
    void* p = tryReuseLargeAlloc(ctx, alignedSize, stream);
    if (p) return p;

    // 2) Unmap if over threshold
    unmapFreeLargeAllocations(ctx, FREE_MAPPED_START_THRESHOLD, FREE_MAPPED_END_THRESHOLD);

    // 3) Try again
    p = allocateNewLargeBlock(ctx, size, stream);
    if (p) return p;

    // 4) If that fails, try fully releasing unused memory then retry
    release_unused_memory_device(ctx.device_id);
    return allocateNewLargeBlock(ctx, size, stream);
}

static void myFreeLarge(DeviceContext& ctx, void* ptr)
{
    if (!ptr) return;
    auto it = ctx.largeAllocRegistry.find(ptr);
    if (it == ctx.largeAllocRegistry.end()) {
        printf("ERROR: Failed to find large allocation to free on device %d\n", ctx.device_id);
        assert(false);
        return;
    }

    LargeAllocInfo info = it->second;
    ctx.largeAllocRegistry.erase(it);
    ctx.largeAllocCache[info.alignedSize].push_back(info);
    ctx.totalMappedFree += info.alignedSize;
}

// ---------------------------------------------------------------------------
// PUBLIC API
// ---------------------------------------------------------------------------
void* sideaware_malloc(size_t size, int device, cudaStream_t stream) {
    debugf("sideaware_malloc(%zu, %d, %p)\n", size, device, stream);

    DeviceContext& ctx = getDeviceContext(device);
    void* p = nullptr;

    if (size >= CUSTOM_ALLOC_THRESHOLD) {
        p = myMallocLarge(ctx, size, stream);
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
    DeviceContext& ctx = getDeviceContext(device);
    ScopedSetDevice guard(device);

    if (size >= CUSTOM_ALLOC_THRESHOLD) {
        myFreeLarge(ctx, ptr);
    } else {
        cudaFreeAsync(ptr, stream);
    }
}

// Assumes device == current device
void sideaware_memcpy(void* dst, const void* src, size_t size, int device, cudaStream_t stream) {
    if (size == 0 || dst == nullptr || src == nullptr) return;
    DeviceContext& ctx = getDeviceContext(device);

    int sm_per_side = ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
    int hash = ctx.cpu_side_info[OFFSET_SIDE_HASH_MASK];
    hash |= (1 << 21); // TODO: make this work with a more complex hash, had issues with bit21+24

    unsigned int size_32b = (unsigned int)size;
    param_sm_side_t &param = ctx.param_sm_side;
    uint4* dst4 = (uint4*)dst;
    uint4* src4 = (uint4*)src;

    // Use the most optimal kernel possible based on 2MiB alignment and whether we need 64-bit indexing or not
    if ((intptr_t)src % (2UL*1024*1024) == 0) {
        if (size < 2UL*1024*1024*1024) {
            side_aware_memcpy<true><<<ctx.num_sms, 1024, 0, stream>>>(dst4, src4, size_32b, hash, sm_per_side, param);
        } else {
            side_aware_memcpy<true><<<ctx.num_sms, 1024, 0, stream>>>(dst4, src4, size, hash, sm_per_side, param);
        }
    } else {
        if (size < 2UL*1024*1024*1024) {
            side_aware_memcpy<false><<<ctx.num_sms, 1024, 0, stream>>>(dst4, src4, size_32b, hash, sm_per_side, param);
        } else {
            side_aware_memcpy<false><<<ctx.num_sms, 1024, 0, stream>>>(dst4, src4, size, hash, sm_per_side, param);
        }
    }
}

// ---------------------------------------------------------------------------
// Additional query/utility/config (mostly per device)
// ---------------------------------------------------------------------------
int get_num_sms() {
    DeviceContext& ctx = getDeviceContext();
    return ctx.num_sms;
}

void fill_sm_sides_tensor(unsigned char* gpu_tensor) {
    DeviceContext& ctx = getDeviceContext();
    cudaError_t err = cudaMemcpy(gpu_tensor, ctx.gpu_side_index, ctx.num_sms, cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);
}

int get_num_sm_side0() {
    DeviceContext& ctx = getDeviceContext();
    return ctx.cpu_side_info[OFFSET_NUM_SM_SIDE0];
}

int get_num_sm_side1() {
    DeviceContext& ctx = getDeviceContext();
    return ctx.cpu_side_info[OFFSET_NUM_SM_SIDE1];
}

int get_min_sm_per_side() {
    DeviceContext& ctx = getDeviceContext();
    return ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
}

int get_hash_mask() {
    DeviceContext& ctx = getDeviceContext();
    return ctx.cpu_side_info[OFFSET_SIDE_HASH_MASK];
}

void set_prealloc_config(int X, int Y) {
    g_preAllocX = X;
    g_preAllocY = Y;
}

// ---------------------------------------------------------------------------
// Release all unused memory for all devices
// ---------------------------------------------------------------------------
size_t release_unused_memory() {
    size_t total_freed = 0;
    for (size_t i = 0; i < g_deviceContexts.size(); i++) {
        if (g_deviceContexts[i].initialized) {
            total_freed += release_unused_memory_device(i);
        }
    }
    return total_freed;
}

} // extern "C"