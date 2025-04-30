// ---------------------------------------------------------------------------------
// Runtime‑compiled side‑aware memcpy kernel collection.
// ---------------------------------------------------------------------------------
#include <cuda_runtime.h>

constexpr int FORCED_UNROLL = 4; // TODO: make easy to configure
constexpr bool FORCE_WRONG_SIDE = false;
constexpr bool FORCE_RANDOM_SIDE = false;
constexpr unsigned int CHUNK_SIZE = 4096;

// must match alloc.cu (TODO: dynamic)
constexpr unsigned int MAX_SM = 200;
constexpr unsigned int NUM_GROUPS = 4;

typedef struct {
    unsigned char side_index[MAX_SM];
} param_sm_side_t;

struct unused {}; // type for inputs/outputs not used in the kernel

// ----------------------------------------------------------------------------
// L2 Side Aware memcpy kernel (single input, single output, any byte count)
// ----------------------------------------------------------------------------
template<typename size_type = size_t>
__device__ __forceinline__ void side_aware_memcpy_device(
        uint4 * __restrict__ output,
        const uint4 * __restrict__ input,
        size_type num_bytes, unsigned int hash, unsigned int num_sm_per_side,
        const param_sm_side_t params) {
    // ...
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

    unsigned int sm_side = params.side_index[smid] & 1;
    unsigned int sm_side_index = params.side_index[smid] >> 1;

    if (sm_side_index >= num_sm_per_side) {
        __nanosleep(1000);
        return;
    }

    unsigned int group = threadIdx.y;
    unsigned int group_tid_offset = threadIdx.x * sizeof(uint4);
    unsigned int num_groups_per_side = num_sm_per_side * NUM_GROUPS;
    unsigned int global_idx = (sm_side_index * NUM_GROUPS) + group;

    unsigned int num_double_chunks = num_bytes / (2 * CHUNK_SIZE);
    unsigned int multi_chunks = num_double_chunks / FORCED_UNROLL;
    unsigned int base = ((unsigned long long)input) & 0xFFFFFFFFULL;

    unsigned int offset_per_j = 2 * CHUNK_SIZE;
    unsigned int offset_per_i = (FORCED_UNROLL * CHUNK_SIZE * 2);
    unsigned int offset_per_i_iter = (offset_per_i * num_groups_per_side) - offset_per_i;
    size_type byte_offset = global_idx * offset_per_i + group_tid_offset;

#pragma unroll
    for (unsigned int i = global_idx; i < multi_chunks; i += num_groups_per_side, byte_offset += offset_per_i_iter) {
        size_type offsets[FORCED_UNROLL];
        uint4 inputs[FORCED_UNROLL];
#pragma unroll FORCED_UNROLL
        for (int j = 0; j < FORCED_UNROLL; j++, byte_offset += offset_per_j) {
            unsigned int lsb_bits = base + (byte_offset & 0xFFFFFFFF);
            unsigned int side = __popc(lsb_bits & hash) & 1;
            if constexpr (FORCE_WRONG_SIDE) side ^= 1;
            if constexpr (FORCE_RANDOM_SIDE) side = 0;

            unsigned int use_second_chunk = sm_side ^ side;
            size_type offset = byte_offset + (use_second_chunk * CHUNK_SIZE);

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
            size_type byte_offset = (size_type)(idx + multi_chunks*FORCED_UNROLL) * (2*CHUNK_SIZE) + group_tid_offset;
            unsigned int lsb_bits = base + (byte_offset & 0xFFFFFFFF);
            unsigned int side = __popc(lsb_bits & hash) & 1;

            unsigned int use_second_chunk = sm_side ^ side;
            byte_offset += use_second_chunk * CHUNK_SIZE;

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
// Multi-Input / Multi-Output Elementwise kernel (generalised version of the above)
// ---------------------------------------------------------------------------------

template<typename size_type = size_t,
         typename o0 = uint4, typename o1 = uint4,
         typename i0 = uint4, typename i1 = uint4, typename i2 = uint4, typename i3 = uint4>
__device__ __forceinline__ void elementwise_op(
        size_type idx,
        o0 &output0, o1 &output1,
        const i0 &input0, const i1 &input1, const i2 &input2, const i3 &input3) {
    // ...
    output0 = input0;
    output1 = input1;
}

template<typename size_type = size_t, bool sideaware_output = false,
        typename o0 = uint4, typename o1 = uint4,
        typename i0 = uint4, typename i1 = uint4, typename i2 = uint4, typename i3 = uint4>
__device__ __forceinline__ void side_aware_elementwise_device(
        o0* __restrict__ output0, o1* __restrict__ output1,
        const i0 * __restrict__ input0, const i1 * __restrict__ input1,
        const i2 * __restrict__ input2, const i3 * __restrict__ input3,
        size_type num_elements, unsigned int hash, unsigned int num_sm_per_side,
        const param_sm_side_t params) {
    // ...
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

    unsigned int sm_side = params.side_index[smid] & 1;
    unsigned int sm_side_index = params.side_index[smid] >> 1;

    if (sm_side_index >= num_sm_per_side) {
        __nanosleep(1000);
        return;
    }

    static_assert(sizeof(i0) == 16 || sizeof(o0) == 16, "i0 and/or o0 must be 16 bytes");
    constexpr size_type element_size = sideaware_output ? sizeof(o0) : sizeof(i0);
    size_type num_bytes = num_elements * element_size;

    unsigned int group = threadIdx.y;
    unsigned int group_tid_offset = threadIdx.x * sizeof(uint4);
    unsigned int num_groups_per_side = num_sm_per_side * NUM_GROUPS;
    unsigned int global_idx = (sm_side_index * NUM_GROUPS) + group;

    unsigned int num_double_chunks = num_bytes / (2 * CHUNK_SIZE);
    unsigned int multi_chunks = num_double_chunks / FORCED_UNROLL;
    unsigned int base = (sideaware_output ? (unsigned long long)output1 : (unsigned long long)input0) & 0xFFFFFFFFULL;

    unsigned int offset_per_j = 2 * CHUNK_SIZE;
    unsigned int offset_per_i = (FORCED_UNROLL * CHUNK_SIZE * 2);
    unsigned int offset_per_i_iter = (offset_per_i * num_groups_per_side) - offset_per_i;
    size_type byte_offset = global_idx * offset_per_i + group_tid_offset;

#pragma unroll
    for (unsigned int i = global_idx; i < multi_chunks; i += num_groups_per_side, byte_offset += offset_per_i_iter) {
        size_type elements[FORCED_UNROLL];
        i0 inputs0[FORCED_UNROLL];
        i1 inputs1[FORCED_UNROLL];
        i2 inputs2[FORCED_UNROLL];
        i3 inputs3[FORCED_UNROLL];
#pragma unroll FORCED_UNROLL
        for (int j = 0; j < FORCED_UNROLL; j++, byte_offset += offset_per_j) {
            unsigned int lsb_bits = base + (byte_offset & 0xFFFFFFFF);

            unsigned int side = __popc(lsb_bits & hash) & 1;
            if constexpr (FORCE_WRONG_SIDE) side ^= 1;
            if constexpr (FORCE_RANDOM_SIDE) side = 0;

            unsigned int use_second_chunk = sm_side ^ side;
            size_type offset = byte_offset + (use_second_chunk * CHUNK_SIZE);

            size_type element = offset / element_size;
            elements[j] = element;
            inputs0[j] = input0[element];
            inputs1[j] = input1[element];
            inputs2[j] = input2[element];
            inputs3[j] = input3[element];
        }
#pragma unroll FORCED_UNROLL
        for (int j = 0; j < FORCED_UNROLL; j++) {
            size_type element = elements[j];
            elementwise_op(element, output0[element], output1[element],
                           inputs0[j], inputs1[j], inputs2[j], inputs3[j]);
        }
    }

    if (group == 0) {
        int max_remaining_double_chunks = (num_double_chunks + 1) - (multi_chunks * FORCED_UNROLL);
        int start_sm_side_idx = num_sm_per_side - max_remaining_double_chunks;
        int idx = sm_side_index - start_sm_side_idx;
        if (idx >= 0) {
            size_type byte_offset = (size_type)(idx + multi_chunks*FORCED_UNROLL) * (2*CHUNK_SIZE) + group_tid_offset;
            unsigned int lsb_bits = base + (byte_offset & 0xFFFFFFFF);
            unsigned int side = __popc(lsb_bits & hash) & 1;

            unsigned int use_second_chunk = sm_side ^ side;
            byte_offset += use_second_chunk * CHUNK_SIZE;

            size_type element = byte_offset / element_size;
            if (element < num_elements) {
                elementwise_op(element, output0[element], output1[element],
                               input0[element], input1[element], input2[element], input3[element]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Explicit wrapper kernels (external names – easier to locate with cuModuleGetFunction)
// ---------------------------------------------------------------------------------

extern "C" {
/*
__global__ void side_aware_elementwise_32(
        uint4* __restrict__ dst, const uint4* __restrict__ src,
        unsigned int num_bytes, unsigned int hash, unsigned int sm_per_side,
        __grid_constant__ const param_sm_side_t params) {

    size_t num_elements = num_bytes / sizeof(uint4);
    side_aware_elementwise_device<unsigned int, false, uint4, unused, uint4, unused, unused, unused>(
        dst, nullptr, src, nullptr, nullptr, nullptr, num_elements, hash, sm_per_side, params);
}
*/

__global__ __launch_bounds__(1024, 1) void side_aware_memcpy_32(
        uint4* __restrict__ dst, const uint4* __restrict__ src,
        unsigned int num_bytes, unsigned int hash, unsigned int sm_per_side,
        __grid_constant__ const param_sm_side_t params) {
    side_aware_memcpy_device<unsigned int>(dst, src, num_bytes, hash, sm_per_side, params);
}

__global__ __launch_bounds__(1024, 1) void side_aware_memcpy_64(
        uint4* __restrict__ dst, const uint4* __restrict__ src,
        size_t num_bytes, unsigned int hash, unsigned int sm_per_side,
        __grid_constant__ const param_sm_side_t params) {
    side_aware_memcpy_device<size_t>(dst, src, num_bytes, hash, sm_per_side, params);
}

} // extern "C"
