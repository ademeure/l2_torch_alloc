#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
// We'll choose 8MiB as the cutoff for small vs. large:
constexpr size_t LARGE_ALLOC_THRESHOLD = 2ULL * 1024ULL * 1024ULL;
// Global free–mapped threshold: if the total free (but mapped) memory exceeds this,
// then unmap blocks until the total falls below FREE_MAPPED_LOWER_THRESHOLD.
constexpr size_t FREE_MAPPED_UPPER_THRESHOLD = 8192ULL * 1024ULL * 1024ULL; // 8GiB
constexpr size_t FREE_MAPPED_LOWER_THRESHOLD = 2048ULL * 1024ULL * 1024ULL; // 2GiB

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define DETERMINE_HASH_DYNAMICALLY
#define DEBUG_PRINTF // used in init only so doesn't affect performance
#ifdef DEBUG_PRINTF
#define debugf(...) printf(__VA_ARGS__)
#else
#define debugf(...)
#endif
#define ceil_div(a, b) (((a) + (b) - 1) / (b))

constexpr size_t CHUNK_SIZE = 4096;            // 4KiB
constexpr size_t PAGE_SIZE  = 2 * 1024 * 1024; // 2MiB
constexpr int L2_SIDE_TEST_ITERATIONS = 25;
constexpr int MAX_SM = 200;
constexpr int OFFSET_ZEROED_COUNTER    = MAX_SM;
constexpr int OFFSET_AVERAGE_LATENCY   = MAX_SM + 1;
constexpr int OFFSET_SIDE_HASH_MASK    = MAX_SM + 2;
constexpr int OFFSET_NUM_SM_SIDE0      = MAX_SM + 3;
constexpr int OFFSET_NUM_SM_SIDE1      = MAX_SM + 4;
constexpr int OFFSET_MIN_SM_PER_SIDE   = MAX_SM + 5;

constexpr bool FORCE_WRONG_SIDE = false;
constexpr bool FORCE_RANDOM_SIDE = false;

// ---------------------------------------------------------------------------
// A single struct that holds everything needed for chunk-based allocation
// ---------------------------------------------------------------------------
struct LargeAllocInfo {
    size_t userRequested   = 0;
    void*  basePtr         = nullptr;  // Points to the correctly aligned section for mapping
    void*  allocPtr        = nullptr;  // Original pointer from cuMemAddressReserve
    size_t alignedSize     = 0;
    bool   useCompression  = false;
    cudaStream_t allocStream = 0;
    // One handle per 2MiB chunk:
    std::vector<CUmemGenericAllocationHandle> handles;

    // NEW: Track which side was actually used for each chunk so we know where to return it:
    std::vector<int> sideUsed; // 0 or 1 for each chunk
};

typedef struct {
    unsigned char side_index[MAX_SM];
} param_sm_side_t;

// Key function defined early as this template function cannot be defined in the extern "C" block
// The wrapper function calling this is defined at the very end (needs access to other variables)
template<typename size_type=int, bool input_aligned_2mib=false, int FORCED_UNROLL=4>
__global__ __launch_bounds__(1024, 1, 1) void side_aware_memcpy(uint4 * __restrict__ output, const uint4 * __restrict__ input, size_type num_bytes, const int hash_mask, const int num_sm_per_side, __grid_constant__ const param_sm_side_t params) {
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

    // Each "group" is 256 threads processing 16 bytes each = 4096 bytes
    int num_groups = blockDim.x / 256;
    int group = threadIdx.x / 256;
    int group_tid = threadIdx.x % 256;
    int group_tid_offset = group_tid * 16;
    int num_groups_per_side = num_sm_per_side * num_groups;
    int global_idx = (sm_side_index * num_groups) + group;

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
            if constexpr (input_aligned_2mib) { lsb_side_bits = byte_offset & hash_mask; }
            else { lsb_side_bits = (input_base + (byte_offset & 0xFFFFFFFF)) & hash_mask; }

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
            //output[offset] = input[offset];
        }
        #pragma unroll FORCED_UNROLL
        for (int j = 0; j < FORCED_UNROLL; j++) {
            if (group_tid == 0) {
                //printf("offsets[j]: %d, blockDim.x: %u\n", (int)sizeof(size_type), blockDim.x);
                //printf("offsets[j]: %u, i: %u, j: %d, threadIdx.x: %u, blockIdx.x: %u, blockDim.x: %u\n", offsets[j], i, j, threadIdx.x, blockIdx.x, blockDim.x);
            }
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
            // Determine the side of the 1st 4KiB chunk in the 8KiB "double chunk"
            int lsb_side_bits = (((input_aligned_2mib ? 0 : input_base) + (byte_offset & 0xFFFFFFFF)) & hash_mask);
            int side = (__popc(lsb_side_bits) & 1);
            if (side != sm_side) {
                byte_offset += CHUNK_SIZE;
            }
            if (byte_offset + sizeof(uint4) <= num_bytes) {
                output[byte_offset / sizeof(uint4)] = input[byte_offset / sizeof(uint4)];
            }
            return;
        }
        // Process the final partial uint4 (when the number of bytes is not a multiple of 16)
        if (sm_side_index == (num_sm_per_side - 1) && sm_side == 0 && (num_bytes % sizeof(uint4) != 0)) {
            size_type byte_offset = threadIdx.x + num_bytes - (num_bytes % sizeof(uint4));
            if(byte_offset < num_bytes) {
                ((unsigned char*)output)[byte_offset] = ((unsigned char*)input)[byte_offset];
            }
        }
    }
}

extern "C" {

// ---------------------------------------------------------------------------
// Global variables
// ---------------------------------------------------------------------------
size_t my_alloc_count = 0;

int num_sms = 0;
int cpu_side_info[MAX_SM * 2];
unsigned char cpu_side_index[MAX_SM];

unsigned int* gpu_allocator_metadata = nullptr;
unsigned int* gpu_side_info              = nullptr; // offset into gpu_allocator_metadata
unsigned char *gpu_side_index;
unsigned char *gpu_scratch_buffer;

constexpr size_t UNBACKED_VIRTUAL_PTR_SIZE = 4096ULL * 1024ULL * 1024ULL; // 4GiB
constexpr size_t UNBACKED_VIRTUAL_PAGES = UNBACKED_VIRTUAL_PTR_SIZE / PAGE_SIZE;
unsigned char cpu_page_side[UNBACKED_VIRTUAL_PAGES];

// Store the desired start side for each allocation size
static std::unordered_map<size_t, int> g_sizeSideMap;

// Global counter for total free mapped memory
static size_t g_totalMappedFree = 0;

// Cached allocation constraints
static bool g_compressionAvailable = false;

// ---------------------------------------------------------------------------
// Device function & kernel for side info
// ---------------------------------------------------------------------------
__device__ __forceinline__ int test_latency_l2(unsigned int* data, size_t offset) {
    unsigned int old_value = atomicExch(&data[offset], 0);
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

#ifdef DETERMINE_HASH_DYNAMICALLY
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
                printf("\nERROR: CHUNK_SIZE %llu not matching lowest hash bit\n\n",
                       (unsigned long long)CHUNK_SIZE);
                assert(false);
            }
#endif
        }
    } else if (threadIdx.x >= 32) {
        __nanosleep(10000);
    }
}

__global__ void test_page_latency(unsigned int* ptr, unsigned int *side_info,
                                  unsigned char* page_side, unsigned int num_pages)
{
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        int near_side = side_info[smid] & 1;
        int far_side  = 1 - near_side;
        int average_latency = side_info[OFFSET_AVERAGE_LATENCY];

        //debugf("Testing %u pages.\n", num_pages);
        for (int i = 0; i < num_pages; i++) {
            size_t offset = (size_t)i * (PAGE_SIZE / sizeof(unsigned int));
            int total_latency = test_latency_l2(ptr, offset);
            page_side[i] = (total_latency > average_latency) ? far_side : near_side;
            /*debugf("[SM %3d] Page %3d: L2-latency = %.1f (raw: %u) -> side=%d\n",
                   smid, i,
                   (float)total_latency / (float)L2_SIDE_TEST_ITERATIONS,
                   total_latency, page_side[i]);*/
        }
    }
}

__global__ void test_single_address_latency(unsigned int* ptr, unsigned int *side_info)
{
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        if(smid != 0) {
            return;
        }
        int total_latency = test_latency_l2(ptr, 0);
        int average_latency = side_info[OFFSET_AVERAGE_LATENCY];
        intptr_t ptr_int = reinterpret_cast<intptr_t>(ptr);
        int expected_side = ((ptr_int >> 21) & 1) ^ ((ptr_int >> 24) & 1);

        printf("total_latency: %d, average_latency: %d ==> %d (expected: %d ===> AGREEMENT: %s)\n", total_latency, average_latency, (total_latency > average_latency), expected_side, (total_latency > average_latency) == expected_side ? "YES" : "NO");
    } else if (threadIdx.x >= 32) {
        __nanosleep(10000);
    }
}

// ---------------------------------------------------------------------------
// Configuration for pre-allocation
// ---------------------------------------------------------------------------
// If (free_handles_side0.size() + free_handles_side1.size()) < needed + X,
// we'll allocate (needed + Y) more handles (split unknown a priori side).
static int g_preAllocX = 1;  // example default
static int g_preAllocY = 10; // example default

// Callers can adjust these if desired.
void setPreAllocationConfig(int X, int Y) {
    g_preAllocX = X;
    g_preAllocY = Y;
}

// ---------------------------------------------------------------------------
// Pre-allocated handle pools (one per side)
// ---------------------------------------------------------------------------
static std::vector<CUmemGenericAllocationHandle> g_freeHandlesSide0;
static std::vector<CUmemGenericAllocationHandle> g_freeHandlesSide1;

// We only warn once if we have to pick from the "wrong" side:
static bool g_crossSideWarningIssued = false;

// ---------------------------------------------------------------------------
// Query the device allocation constraints
// ---------------------------------------------------------------------------
static void init_allocation_constraints()
{
    int comp_available;
    cuDeviceGetAttribute(&comp_available, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, 0);
    g_compressionAvailable = (comp_available != 0);

    // Verify granularity equals PAGE_SIZE
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t granularity;
    CUresult res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(res == CUDA_SUCCESS && granularity == PAGE_SIZE);

    if (g_compressionAvailable) {
        prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        assert(res == CUDA_SUCCESS && granularity == PAGE_SIZE);
    }
}

static CUmemAllocationProp get_allocation_constraints(bool use_compression=false)
{
    use_compression = use_compression && g_compressionAvailable;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.allocFlags.compressionType = use_compression ? CU_MEM_ALLOCATION_COMP_GENERIC : 0;

    return prop;
}

// ---------------------------------------------------------------------------
// We still keep caches for large blocks (mapped) + registry of active blocks
// ---------------------------------------------------------------------------
static std::unordered_map<size_t, std::vector<LargeAllocInfo>> g_largeAllocCache;
static std::unordered_map<void*, LargeAllocInfo> g_largeAllocRegistry;

// Forward declarations
static CUresult allocateCompressible(LargeAllocInfo &info, size_t size, bool use_compression=false);
size_t releaseUnusedMemory(); // used internally

// ---------------------------------------------------------------------------
// Unmap free blocks if threshold is exceeded
// (unchanged from original, except we no longer push handles into a single
//  'g_reusableHandles' vector but call cuMemRelease directly or keep them separate.)
// ---------------------------------------------------------------------------
static void unmapFreeLargeAllocations(size_t start_threshold=0,
                                      size_t end_threshold=0) {
    if (g_totalMappedFree <= start_threshold) return;
    cudaDeviceSynchronize();

    for (auto &kv : g_largeAllocCache) {
        auto &vec = kv.second;
        auto it = vec.begin();

        while (it != vec.end() && g_totalMappedFree > end_threshold) {
            LargeAllocInfo &blk = *it;
            size_t size = blk.alignedSize;
            size_t num_pages = blk.alignedSize / PAGE_SIZE;
            CUdeviceptr base = (CUdeviceptr)blk.basePtr;

            for (size_t i = 0; i < num_pages; i++) {
                cuMemUnmap(base + i * PAGE_SIZE, PAGE_SIZE);
                // Return handle to whichever side it was allocated from:
                int side = blk.sideUsed[i];
                if (side == 0) {
                    g_freeHandlesSide0.push_back(blk.handles[i]);
                } else {
                    g_freeHandlesSide1.push_back(blk.handles[i]);
                }
            }
            // Use allocPtr for freeing address space, not basePtr
            cuMemAddressFree((CUdeviceptr)blk.allocPtr, blk.alignedSize + 2 * PAGE_SIZE);

            assert(g_totalMappedFree >= size);
            g_totalMappedFree -= size;
            it = vec.erase(it);
        }
        if (g_totalMappedFree <= end_threshold)
            break;
    }
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Pre-allocate new physical handles (for whichever side they turn out to be)
// by actually mapping them into a temporary VA range, running test_page_latency,
// and storing the result in g_freeHandlesSide0 or g_freeHandlesSide1.
// ---------------------------------------------------------------------------
static CUresult preAllocateHandles(size_t countNeeded, bool useCompression, cudaStream_t stream)
{
    if (countNeeded == 0) return CUDA_SUCCESS;

    CUmemAllocationProp prop = get_allocation_constraints(useCompression);
    CUresult lastError = CUDA_SUCCESS;
    size_t totalAllocated = 0;

    // We'll do it in batches so we don't exceed UNBACKED_VIRTUAL_PAGES at once.
    while (countNeeded > 0) {
        size_t batch = std::min<size_t>(countNeeded, UNBACKED_VIRTUAL_PAGES);
        size_t batchAllocated = 0;

        // Reserve a temporary VA space for `batch` pages:
        CUdeviceptr dptr;
        lastError = cuMemAddressReserve(&dptr, batch * PAGE_SIZE, 0, 0, 0);
        if (lastError != CUDA_SUCCESS) {
            break; // Can't continue this batch, but process what we have so far
        }

        // Create + map each handle:
        std::vector<CUmemGenericAllocationHandle> tempHandles(batch);
        for (size_t i = 0; i < batch; i++) {
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
                batch = i; // adjust batch size
                break;
            }
            batchAllocated++;
        }

        // If we allocated nothing in this batch, clean up and continue
        if (batchAllocated == 0) {
            cuMemAddressFree(dptr, batch * PAGE_SIZE);
            break;
        }

        // Set access for whatever we allocated
        CUmemAccessDesc accessDesc;
        accessDesc.location.id   = prop.location.id;
        accessDesc.location.type = prop.location.type;
        accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        lastError = cuMemSetAccess(dptr, batchAllocated * PAGE_SIZE, &accessDesc, 1);
        if (lastError != CUDA_SUCCESS) {
            for (size_t i = 0; i < batchAllocated; i++) {
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
            // This should never happen since we managed to map everything etc...
            printf("ERROR: Failed to zero out memory before test\n");
            assert(false);
            return lastError;
        }

        // Now classify each page by calling test_page_latency:
        test_page_latency<<<1, 512, 0, stream>>>(
            reinterpret_cast<unsigned int*>(dptr), gpu_side_info, gpu_scratch_buffer, (unsigned int)batchAllocated);
        cudaMemcpy(cpu_page_side, gpu_scratch_buffer, batchAllocated, cudaMemcpyDeviceToHost);

        // Unmap + place each handle in correct side's pool
        for (size_t i = 0; i < batchAllocated; i++) {
            cuMemUnmap(dptr + i * PAGE_SIZE, PAGE_SIZE);
            int side = cpu_page_side[i];
            if (side == 0) {
                g_freeHandlesSide0.push_back(tempHandles[i]);
            } else {
                g_freeHandlesSide1.push_back(tempHandles[i]);
            }
        }
        cuMemAddressFree(dptr, batch * PAGE_SIZE);

        totalAllocated += batchAllocated;
        countNeeded -= batchAllocated;

        // If we couldn't allocate the full batch, we're likely out of resources
        if (batchAllocated < batch) {
            break;
        }
    }

    // If we allocated anything at all, consider it at least a partial success
    if (totalAllocated > 0) {
        return CUDA_SUCCESS;
    }

    // Otherwise return the last error we encountered
    return lastError;
}

// ---------------------------------------------------------------------------
// Ensure we have enough free handles (side0+side1) for the upcoming allocation.
// If freeHandles < needed + X, then allocate (needed + Y) more handles overall.
// ---------------------------------------------------------------------------
static CUresult ensureFreeHandlesAvailable(size_t needed, bool useCompression, cudaStream_t stream)
{
    size_t needed_per_side = (needed + g_preAllocX) / 2;
    size_t free0 = g_freeHandlesSide0.size();
    size_t free1 = g_freeHandlesSide1.size();
    size_t needed_side0 = (needed_per_side > free0) ? (needed_per_side - free0) : 0;
    size_t needed_side1 = (needed_per_side > free1) ? (needed_per_side - free1) : 0;
    size_t needed_worst_case = max(needed_side0, needed_side1);
    size_t needed_both_sides = 2 * needed_worst_case;

    // Print EVERYTHING for debugging purposes
    //printf("needed: %zu, needed_per_side: %zu, needed_side0: %zu, needed_side1: %zu, needed_worst_case: %zu, needed_both_sides: %zu\n",
    //       needed, needed_per_side, needed_side0, needed_side1, needed_worst_case, needed_both_sides);
    //printf("g_freeHandlesSide0.size(): %zu, g_freeHandlesSide1.size(): %zu\n",
    //       g_freeHandlesSide0.size(), g_freeHandlesSide1.size());

    if (needed_both_sides > 0) {
        needed_both_sides += (g_preAllocY - g_preAllocX);
        CUresult rc = preAllocateHandles(needed_both_sides, useCompression, stream);
        return rc;
    }
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// Decide which side to pick based on bits 21 and 24 of the VA
// ---------------------------------------------------------------------------
static inline int pickSideFromVA(uint64_t va)
{
    // TODO: Make this work with a more complex hash, had issues with bit21+24

    // bits 21 & 24 => shift right 21 => get bit0
    //                shift right 24 => get bit1
    // We can do a quick check:
    // if ((va >> 21) & 1) == ((va >> 24) & 1) => side0 else side1
    int bit21 = (int)((va >> 21) & 1ULL);
    return bit21;
    //int bit24 = (int)((va >> 24) & 1ULL);
    //return (bit21 == bit24) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// The new "allocateCompressible" function: it no longer calls cuMemCreate
// inside the loop. Instead, it relies on ensureFreeHandlesAvailable() to have
// pre-allocated enough handles in g_freeHandlesSide0/g_freeHandlesSide1.
// Then it picks from the correct side according to bits 21 and 24 of the VA.
// ---------------------------------------------------------------------------
static CUresult allocateCompressible(LargeAllocInfo &info, size_t size, bool use_compression)
{
    info.alignedSize = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    info.useCompression = (use_compression && g_compressionAvailable);

    // 1) Pre-check we have enough free handles, otherwise allocate more:
    size_t num_pages = info.alignedSize / PAGE_SIZE;
    CUresult rc = ensureFreeHandlesAvailable(num_pages, info.useCompression, info.allocStream);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    // Determine the desired "start side" for this size
    // If we haven't seen this size before, choose the side with fewer handles
    if (g_sizeSideMap.find(info.alignedSize) == g_sizeSideMap.end()) {
        int side0_free = g_freeHandlesSide0.size();
        int side1_free = g_freeHandlesSide1.size();
        g_sizeSideMap[info.alignedSize] = (side0_free >= side1_free) ? 0 : 1;
    }
    int desiredStartSide = g_sizeSideMap[info.alignedSize];

    // 2) Reserve VA space with EXTRA space (2 extra pages = 4MiB)
    // This ensures we'll find a properly-aligned section with the desired side
    CUdeviceptr allocPtr = 0;
    size_t extraSize = info.alignedSize + 2 * PAGE_SIZE; // Add 4MiB extra space
    rc = cuMemAddressReserve(&allocPtr, extraSize, 0, 0, 0);
    if (rc != CUDA_SUCCESS) {
        return rc; // Fatal error, can't reserve VA
    }

    // Store the original allocation pointer
    info.allocPtr = reinterpret_cast<void*>(allocPtr);

    // Find the starting point for the desired side
    CUdeviceptr basePtr = allocPtr;

    // Determine the side of the first 2MiB page
    int firstPageSide = pickSideFromVA((uint64_t)basePtr);

    // If the first page is not on the desired side, try the second page
    if (firstPageSide != desiredStartSide) {
        basePtr += PAGE_SIZE; // Skip to next 2MiB page
    }

    // Verify that this page or the 3rd page has the desired side
    int basePageSide = pickSideFromVA((uint64_t)basePtr);
    if (basePageSide != desiredStartSide) {
        basePtr += PAGE_SIZE;
        int thirdPageSide = pickSideFromVA((uint64_t)basePtr);
        if (thirdPageSide != desiredStartSide) {
            // This should never happen with our 2-bit hash function (at the time of writing)
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
            (desiredSide == 0) ? &g_freeHandlesSide0 : &g_freeHandlesSide1;
        std::vector<CUmemGenericAllocationHandle>* otherPool   =
            (desiredSide == 0) ? &g_freeHandlesSide1 : &g_freeHandlesSide0;

        if (!correctPool->empty()) {
            info.handles[i] = correctPool->back();
            correctPool->pop_back();
            info.sideUsed[i] = desiredSide;
        } else {
            // fallback to the other side
            if (otherPool->empty()) {
                // Very unusual if both are empty – maybe out of memory?
                cuMemAddressFree(allocPtr, extraSize);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            // Warn only the very first time (per the requirement)
            if (!g_crossSideWarningIssued) {
                printf("WARNING: Cross-side handle usage!\n");
                g_crossSideWarningIssued = true;
            }
            info.handles[i] = otherPool->back();
            otherPool->pop_back();
            info.sideUsed[i] = desiredSide;
        }

        rc = cuMemMap(basePtr + i * PAGE_SIZE, PAGE_SIZE, 0, info.handles[i], 0);
        if (rc != CUDA_SUCCESS) {
            // Return the handle to whichever side we took it from:
            if (info.sideUsed[i] == 0) {
                g_freeHandlesSide0.push_back(info.handles[i]);
            } else {
                g_freeHandlesSide1.push_back(info.handles[i]);
            }
            // cleanup
            for (size_t j = 0; j < i; j++) {
                cuMemUnmap(basePtr + j * PAGE_SIZE, PAGE_SIZE);
                // Return that handle
                if (info.sideUsed[j] == 0) {
                    g_freeHandlesSide0.push_back(info.handles[j]);
                } else {
                    g_freeHandlesSide1.push_back(info.handles[j]);
                }
            }
            cuMemAddressFree(allocPtr, extraSize);
            return rc;
        }
    }

    // 4) Set access
    CUmemAccessDesc accessDesc;
    accessDesc.location.id   = 0;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    rc = cuMemSetAccess(basePtr, info.alignedSize, &accessDesc, 1);
    if (rc != CUDA_SUCCESS) {
        // cleanup
        for (size_t i = 0; i < num_pages; i++) {
            cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
            // Return handle
            if (info.sideUsed[i] == 0) g_freeHandlesSide0.push_back(info.handles[i]);
            else                      g_freeHandlesSide1.push_back(info.handles[i]);
        }
        cuMemAddressFree(allocPtr, extraSize);
        return rc;
    }

    // 5) Zero memory to be safe
    cudaDeviceSynchronize();
    rc = cuMemsetD8(basePtr, 0, info.alignedSize);
    if (rc != CUDA_SUCCESS) {
        // cleanup
        for (size_t i = 0; i < num_pages; i++) {
            cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
            // Return handle
            if (info.sideUsed[i] == 0) g_freeHandlesSide0.push_back(info.handles[i]);
            else                      g_freeHandlesSide1.push_back(info.handles[i]);
        }
        cuMemAddressFree(allocPtr, extraSize);
        return rc;
    }

    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// LargeAlloc: Try to reuse from g_largeAllocCache or allocate new
// ---------------------------------------------------------------------------
static void* tryReuseLargeAlloc(size_t alignedSize, cudaStream_t currentStream)
{
    auto &vec = g_largeAllocCache[alignedSize];
    if (vec.empty()) return nullptr;

    LargeAllocInfo info = vec.back();
    assert(g_totalMappedFree >= info.alignedSize);
    g_totalMappedFree -= info.alignedSize;
    vec.pop_back();

    // If different stream, might need synchronization
    if (info.allocStream != currentStream) {
        info.allocStream = currentStream;
        cudaDeviceSynchronize();
    }
    // Insert back into active registry
    g_largeAllocRegistry[info.basePtr] = info;
    return info.basePtr;
}

static void* allocateNewLargeBlock(size_t userSize, cudaStream_t stream)
{
    LargeAllocInfo info;
    info.userRequested = userSize;
    info.allocStream = stream;
    CUresult rc = allocateCompressible(info, userSize);
    if (rc != CUDA_SUCCESS)
        return nullptr;
    g_largeAllocRegistry[info.basePtr] = info;
    return info.basePtr;
}

static void* myMallocLarge(size_t size, cudaStream_t stream)
{
    size_t alignedSize = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

    // 1) Try exact-size reuse
    void* p = tryReuseLargeAlloc(alignedSize, stream);
    if (p) return p;

    // 2) Unmap if over threshold
    unmapFreeLargeAllocations(FREE_MAPPED_UPPER_THRESHOLD, FREE_MAPPED_LOWER_THRESHOLD);

    // 3) Attempt new
    p = allocateNewLargeBlock(size, stream);
    if (p) return p;

    // 4) If that fails, try releasing and retry
    releaseUnusedMemory();
    return allocateNewLargeBlock(size, stream);
}

static void myFreeLarge(void* ptr)
{
    if (!ptr) return;
    auto it = g_largeAllocRegistry.find(ptr);
    if (it == g_largeAllocRegistry.end()) {
        // TODO: in the future, we could take this as meaning we used cudaMallocAsync instead
        // and just call cudaFreeAsync here (which will error if that's not the case, I think? so still works)
        printf("ERROR: Failed to find large allocation to free\n");
        assert(false);
        return;
    }
    LargeAllocInfo info = it->second;
    g_largeAllocRegistry.erase(it);
    g_largeAllocCache[info.alignedSize].push_back(info);
    g_totalMappedFree += info.alignedSize;
}

// ---------------------------------------------------------------------------
// PUBLIC API
// ---------------------------------------------------------------------------
void* my_malloc(size_t size, int device, cudaStream_t stream) {
    static bool firstCall = true;
    if (firstCall) {
        firstCall = false;
        cudaSetDevice(device);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

        init_allocation_constraints();
        cudaMalloc(&gpu_allocator_metadata, PAGE_SIZE);
        assert((uintptr_t)gpu_allocator_metadata % PAGE_SIZE == 0);
        gpu_side_info = &gpu_allocator_metadata[16 * 1024];
        gpu_side_index = (unsigned char*)&gpu_allocator_metadata[32 * 1024];
        gpu_scratch_buffer = (unsigned char*)&gpu_allocator_metadata[128 * 1024];

        cudaDeviceSynchronize();
        init_side_info<<<num_sms, 512, 0, stream>>>(gpu_allocator_metadata, gpu_side_info, gpu_side_index);
        cudaMemcpyAsync(cpu_side_index, gpu_side_index, MAX_SM * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
        // non-async, so forces completion of previous cudaMemcpyAsync
        cudaMemcpy(cpu_side_info, gpu_side_info, MAX_SM * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    my_alloc_count++;

    if (size >= LARGE_ALLOC_THRESHOLD) {
        void* p = myMallocLarge(size, stream);
        // Check side using pickSideFromVA
        /*if (p) {
            int side = pickSideFromVA((uint64_t)p);
            printf("Allocated %zu bytes on side %d\n", size, side);
            test_single_address_latency<<<num_sms, 512, 0, stream>>>((unsigned int*)p, gpu_side_info);
        }*/
        return p;
    } else {
        void* p = nullptr;
        cudaError_t err = cudaMallocAsync(&p, size, stream);
        if (err == cudaSuccess) return p;

        unmapFreeLargeAllocations();
        err = cudaMallocAsync(&p, size, stream);
        if (err == cudaSuccess) return p;
        return nullptr;
    }
}

void my_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (!ptr) return;

    if (size >= LARGE_ALLOC_THRESHOLD) {
        myFreeLarge(ptr);
    } else {
        cudaFreeAsync(ptr, stream);
    }
}

// ---------------------------------------------------------------------------
// Additional query/utility
// ---------------------------------------------------------------------------
int get_num_sms() {
    if (num_sms <= 0) {
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    }
    return num_sms;
}

void fill_sm_sides_tensor(unsigned char* gpu_tensor) {
    assert(my_alloc_count > 0);
    cudaError_t err = cudaMemcpy(gpu_tensor, gpu_side_index, num_sms, cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);
}
int get_num_sm_side0()     { return cpu_side_info[OFFSET_NUM_SM_SIDE0]; }
int get_num_sm_side1()     { return cpu_side_info[OFFSET_NUM_SM_SIDE1]; }
int get_min_sm_per_side()  { return cpu_side_info[OFFSET_MIN_SM_PER_SIDE]; }
int get_hash_mask()        { return cpu_side_info[OFFSET_SIDE_HASH_MASK]; }

// ---------------------------------------------------------------------------
// Release all unused memory (unmap + release handles)
// ---------------------------------------------------------------------------
size_t releaseUnusedMemory() {
    size_t freed_memory = 0;

    // 1) Unmap all cached blocks
    unmapFreeLargeAllocations();

    // 2) Release handles from side0/side1 pools
    for (auto &h : g_freeHandlesSide0) {
        cuMemRelease(h);
        freed_memory += PAGE_SIZE;
    }
    g_freeHandlesSide0.clear();

    for (auto &h : g_freeHandlesSide1) {
        cuMemRelease(h);
        freed_memory += PAGE_SIZE;
    }
    g_freeHandlesSide1.clear();

    cudaDeviceSynchronize();
    return freed_memory;
}

// C API function for manual memcpy
void sideaware_memcpy(void* dst, const void* src, size_t size, cudaStream_t stream) {
    if (size == 0 || dst == nullptr || src == nullptr) return;

    int sm_per_side = cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
    int hash_mask = cpu_side_info[OFFSET_SIDE_HASH_MASK];
    hash_mask |= (1 << 21); // TODO: make this work with a more complex hash, had issues with bit21+24
    param_sm_side_t params;
    for (int i = 0; i < MAX_SM; i++) {
        params.side_index[i] = cpu_side_index[i];
    }

    side_aware_memcpy<<<num_sms, 1024, 0, stream>>>((uint4*)dst, (uint4*)src, size, hash_mask, sm_per_side, params);
}

} // extern "C"