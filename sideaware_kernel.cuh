// ----------------------------------------------------------------------------
// L2 Side Aware Multi-Input / Multi-Output Elementwise kernel (16B alignment)
// ----------------------------------------------------------------------------
#include <cuda_runtime.h>

#ifndef KERNEL_NAME
#define KERNEL_NAME side_aware_elementwise
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