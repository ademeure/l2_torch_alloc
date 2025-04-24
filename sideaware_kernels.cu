// ---------------------------------------------------------------------------------
// Runtime‑compiled side‑aware memcpy kernel collection.
// ---------------------------------------------------------------------------------
#include <cuda_runtime.h>
constexpr bool FORCE_RANDOM_SIDE = false;
constexpr bool FORCE_WRONG_SIDE = false;
constexpr int CHUNK_SIZE = 4096;
constexpr int MAX_SM = 200; // must match alloc.cu

typedef struct {
    unsigned char side_index[MAX_SM];
} param_sm_side_t;

// ---------------------------------------------------------------------------------
// Full side‑optimized memcpy kernel (copied from original alloc.cu template).
// ---------------------------------------------------------------------------------

template<bool input_aligned_2mib=false, int FORCED_UNROLL=4, typename size_type=size_t>
__device__ __forceinline__ void side_aware_memcpy_device(
        uint4 * __restrict__ output,
        const uint4 * __restrict__ input,
        size_type num_bytes,
        int hash,
        int num_sm_per_side,
        const param_sm_side_t params) {

    int smid;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

    int sm_side = params.side_index[smid] & 1;
    int sm_side_index = params.side_index[smid] >> 1;

    if (sm_side_index >= num_sm_per_side) {
        __nanosleep(1000);
        return;
    }

    unsigned int num_groups = blockDim.x / 256;
    unsigned int group = threadIdx.x / 256;
    unsigned int group_tid = threadIdx.x % 256;
    unsigned int group_tid_offset = group_tid * 16;
    unsigned int num_groups_per_side = num_sm_per_side * num_groups;
    unsigned int global_idx = (sm_side_index * num_groups) + group;

    unsigned int num_double_chunks = num_bytes / (2 * CHUNK_SIZE);
    unsigned int multi_chunks = num_double_chunks / FORCED_UNROLL;
    unsigned int input_base = (reinterpret_cast<unsigned long long>(input)) & 0xFFFFFFFFULL;

    unsigned int offset_per_j = 2 * CHUNK_SIZE;
    unsigned int offset_per_i = (FORCED_UNROLL * CHUNK_SIZE * 2);
    unsigned int offset_per_i_iter = (offset_per_i * num_groups_per_side) - offset_per_i;
    size_type byte_offset = global_idx * offset_per_i + group_tid_offset;

    if constexpr (input_aligned_2mib) {
        // we are 2MiB aligned but not necessarily e.g. 4MiB aligned for the start address
        // we use bit 21 in the hash with our custom allocator so need to compensate for this
        sm_side ^= __popc(input_base & hash) & 1;
    }

#pragma unroll
    for (unsigned int i = global_idx; i < multi_chunks; i += num_groups_per_side, byte_offset += offset_per_i_iter) {
        size_type offsets[FORCED_UNROLL];
        uint4 inputs[FORCED_UNROLL];
#pragma unroll FORCED_UNROLL
        for (int j = 0; j < FORCED_UNROLL; j++, byte_offset += offset_per_j) {
            int lsb_side_bits;
            if constexpr (input_aligned_2mib) {
                lsb_side_bits = byte_offset & hash;
            } else {
                lsb_side_bits = (input_base + (byte_offset & 0xFFFFFFFF)) & hash;
            }

            int side = __popc(lsb_side_bits) & 1;
            if constexpr (FORCE_RANDOM_SIDE) {
                side = 0;
            } else if constexpr (FORCE_WRONG_SIDE) {
                side ^= 1;
            }

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

    if (group == 0) {
        int max_remaining_double_chunks = (num_double_chunks + 1) - (multi_chunks * FORCED_UNROLL);
        int start_sm_side_idx = num_sm_per_side - max_remaining_double_chunks;
        int idx = sm_side_index - start_sm_side_idx;
        if (idx >= 0) {
            size_type byte_offset = (size_type)(idx + multi_chunks * FORCED_UNROLL) * (2 * CHUNK_SIZE) + group_tid * 16;
            int lsb_side_bits = (((input_aligned_2mib ? 0 : input_base) + (byte_offset & 0xFFFFFFFF)) & hash);
            int side = (__popc(lsb_side_bits) & 1);
            if (side != sm_side) {
                byte_offset += CHUNK_SIZE;
            }
            if (byte_offset + sizeof(uint4) <= num_bytes) {
                output[byte_offset / sizeof(uint4)] = input[byte_offset / sizeof(uint4)];
            }
        } else if (idx == -1 && sm_side == 0) {
            size_type byte_offset = threadIdx.x + num_bytes - (num_bytes % sizeof(uint4));
            if(byte_offset < num_bytes) {
                ((unsigned char*)output)[byte_offset] = ((unsigned char*)input)[byte_offset];
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Explicit wrapper kernels (external names – easier to locate with cuModuleGetFunction)
// ---------------------------------------------------------------------------------

extern "C" {

__global__ void side_aware_memcpy_aligned_32(
        uint4* __restrict__ dst, const uint4* __restrict__ src,
        unsigned int num_bytes, int hash, int sm_per_side, __grid_constant__ const param_sm_side_t params) {
    side_aware_memcpy_device<true, 4, unsigned int>(dst, src, num_bytes, hash, sm_per_side, params);
}

__global__ void side_aware_memcpy_aligned_64(
        uint4* __restrict__ dst, const uint4* __restrict__ src,
        size_t num_bytes, int hash, int sm_per_side, __grid_constant__ const param_sm_side_t params) {
    side_aware_memcpy_device<true, 4, size_t>(dst, src, num_bytes, hash, sm_per_side, params);
}

__global__ void side_aware_memcpy_unaligned_32(
        uint4* __restrict__ dst, const uint4* __restrict__ src,
        unsigned int num_bytes, int hash, int sm_per_side, __grid_constant__ const param_sm_side_t params) {
    side_aware_memcpy_device<false, 4, unsigned int>(dst, src, num_bytes, hash, sm_per_side, params);
}

__global__ void side_aware_memcpy_unaligned_64(
        uint4* __restrict__ dst, const uint4* __restrict__ src,
        size_t num_bytes, int hash, int sm_per_side, __grid_constant__ const param_sm_side_t params) {
    side_aware_memcpy_device<false, 4, size_t>(dst, src, num_bytes, hash, sm_per_side, params);
}

} // extern "C"
