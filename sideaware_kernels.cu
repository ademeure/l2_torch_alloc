// ---------------------------------------------------------------------------------
// Runtime‑compiled side‑aware memcpy kernel collection.
// ---------------------------------------------------------------------------------
#include <cuda_runtime.h>

// must match alloc.cu (TODO: dynamic)
constexpr unsigned int HASH = 0x2B3000;
constexpr unsigned int MAX_SM = 200;
constexpr unsigned int NUM_GROUPS = 4;
constexpr bool FORCE_WRONG_SIDE = false;
constexpr bool FORCE_RANDOM_SIDE = false;

constexpr unsigned int CHUNK_SIZE = 4096;

template<typename T, typename U>
struct is_same { static constexpr bool value = false; };
template<typename T>
struct is_same<T, T> { static constexpr bool value = true; };

typedef struct {
    unsigned char side_index[MAX_SM];
} param_sm_side_t;

struct unused {}; // type for inputs/outputs not used in the kernel

template<typename T, size_t N>
struct __align__(16) packed {
    T data[N];
    __device__ __forceinline__ T& operator[](int index) { return data[index]; }
    __device__ __forceinline__ const T& operator[](int index) const { return data[index]; }
};

template<bool evict = false, typename T, size_t N>
__device__ __forceinline__ packed<T, N> load(const packed<T, N> * __restrict__ src) {
    if constexpr (evict) {
        if constexpr (sizeof(T) * N == 16) {
            int4 data = __ldcs((const int4*)src);
            return *(packed<T, N>*)&data;
        } else if constexpr (sizeof(T) * N == 8) {
            int2 data = __ldcs((const int2*)src);
            return *(packed<T, N>*)&data;
        } else if constexpr (sizeof(T) * N == 4) {
            int data = __ldcs((const int*)src);
            return *(packed<T, N>*)&data;
        }
    }
    return *src;
}

typedef unsigned int o0;
typedef unused o1;
typedef unsigned int i0;
typedef unused i1;
typedef unused i2;
typedef unused i3;
constexpr bool input_evict[4] = {true, true, true, true};
constexpr bool input_discard[4] = {false, false, false, false};

constexpr bool reverse_order = true;
constexpr bool sideaware_for_o0 = false;
constexpr int parallel_iterations = 4;
constexpr int vec_size = 4;

typedef packed<o0, vec_size> vo0;
typedef packed<o1, vec_size> vo1;
typedef packed<i0, vec_size> vi0;
typedef packed<i1, vec_size> vi1;
typedef packed<i2, vec_size> vi2;
typedef packed<i3, vec_size> vi3;

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

// ---------------------------------------------------------------------------------
// L2 Side Aware Multi-Input / Multi-Output Elementwise kernel (requires 16B alignment)
// ---------------------------------------------------------------------------------

template<typename size_type = long long>
__device__ __forceinline__ void elementwise_op(
        size_type element_idx, o0 &output0, o1 &output1,
        const i0 &input0, const i1 &input1, const i2 &input2, const i3 &input3) {
    // ...
    output0 = (o0)(input0);
    output1 = (o1)(input1);
}

template<typename size_type = long long>
__device__ __forceinline__ void vector_op(
        size_type vec_idx, vo0 &output0, vo1 &output1,
        const vi0 &input0, const vi1 &input1, const vi2 &input2, const vi3 &input3) {
    // ...
    for (int i = 0; i < vec_size; i++) {
        elementwise_op(vec_idx * vec_size + i, output0[i], output1[i], input0[i], input1[i], input2[i], input3[i]);
    }
}

template<typename size_type = long long>
__device__ __forceinline__ void side_aware_elementwise_device(
        vo0* __restrict__ output0, vo1* __restrict__ output1,
        const vi0 * __restrict__ input0, const vi1 * __restrict__ input1,
        const vi2 * __restrict__ input2, const vi3 * __restrict__ input3,
        size_type num_elements, unsigned int num_sm_per_side,
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

    // blockDim.x * element_size must equal CHUNK_SIZE (we check this on the host side)
    // e.g. 256 threads * 16 bytes = 4096 bytes
    constexpr size_type element_size = sideaware_for_o0 ? sizeof(vo0) : sizeof(vi0);

    int group = threadIdx.y;
    int group_tid_offset = threadIdx.x * element_size;
    int num_groups_per_side = num_sm_per_side * NUM_GROUPS;

    int num_double_chunks = (num_elements * element_size) / (2 * CHUNK_SIZE);
    int multi_chunks = num_double_chunks / parallel_iterations;

    size_type offset_per_group = (parallel_iterations * CHUNK_SIZE * 2);
    size_type offset_outer_loop = (offset_per_group * num_groups_per_side);
    offset_outer_loop = (reverse_order ? -offset_outer_loop : offset_outer_loop) - offset_per_group;

    int global_idx = (sm_side_index * NUM_GROUPS) + group;
    int adjusted_global_idx = reverse_order ? (multi_chunks - global_idx - 1) : global_idx;
    size_type byte_offset = adjusted_global_idx * offset_per_group + group_tid_offset;
    unsigned int base = (sideaware_for_o0 ? (unsigned long long)output0 : (unsigned long long)input0) & 0xFFFFFFFFULL;

    size_type elements[parallel_iterations];
    vi0 inputs0[parallel_iterations];
    vi1 inputs1[parallel_iterations];
    vi2 inputs2[parallel_iterations];
    vi3 inputs3[parallel_iterations];
    vo0 out0;
    vo1 out1;
    int j;

    #pragma unroll 1
    for (int i = global_idx; i < multi_chunks; i += num_groups_per_side, byte_offset += offset_outer_loop) {
        #pragma unroll parallel_iterations
        for (j = 0; j < parallel_iterations; j++, byte_offset += 2*CHUNK_SIZE) {
            unsigned int lsb_bits = base + (byte_offset & 0xFFFFFFFF);

            unsigned int side = __popc(lsb_bits & HASH) & 1;
            if constexpr (FORCE_WRONG_SIDE) side ^= 1;
            if constexpr (FORCE_RANDOM_SIDE) side = 0;

            unsigned int use_second_chunk = sm_side ^ side;
            size_type offset = byte_offset + (use_second_chunk * CHUNK_SIZE);

            size_type element = offset / element_size;
            elements[j] = element;
            inputs0[j] = load<input_evict[0]>(&input0[element]);
            inputs1[j] = load<input_evict[1]>(&input1[element]);
            inputs2[j] = load<input_evict[2]>(&input2[element]);
            inputs3[j] = load<input_evict[3]>(&input3[element]);
        }

        #pragma unroll parallel_iterations
        for (j = 0; j < parallel_iterations; j++) {
            size_type element = elements[j];
            discard_inputs(input0 + element, input1 + element, input2 + element, input3 + element);
        }

        #pragma unroll parallel_iterations
        for (j = 0; j < parallel_iterations; j++) {
            size_type element = elements[j];
            vector_op(element, out0, out1, inputs0[j], inputs1[j], inputs2[j], inputs3[j]);
            output0[element] = out0;
            output1[element] = out1;
        }
    }

    if (group == 0) {
        int max_remaining_double_chunks = (num_double_chunks + 1) - (multi_chunks * parallel_iterations);
        int start_sm_side_idx = num_sm_per_side - max_remaining_double_chunks;
        int idx = sm_side_index - start_sm_side_idx;
        if (idx >= 0) {
            size_type byte_offset = (size_type)(idx + multi_chunks*parallel_iterations) * (2*CHUNK_SIZE) + group_tid_offset;
            unsigned int lsb_bits = base + (byte_offset & 0xFFFFFFFF);
            unsigned int side = __popc(lsb_bits & HASH) & 1;

            unsigned int use_second_chunk = sm_side ^ side;
            byte_offset += use_second_chunk * CHUNK_SIZE;

            size_type element = byte_offset / element_size;
            if (element < num_elements) {
                j = 0;
                inputs0[j] = load<input_evict[0]>(input0 + element);
                inputs1[j] = load<input_evict[1]>(input1 + element);
                inputs2[j] = load<input_evict[2]>(input2 + element);
                inputs3[j] = load<input_evict[3]>(input3 + element);
                discard_inputs(input0 + element, input1 + element, input2 + element, input3 + element);

                vector_op(element, out0, out1, inputs0[j], inputs1[j], inputs2[j], inputs3[j]);
                output0[element] = out0;
                output1[element] = out1;
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Explicit wrapper kernels (external names – easier to locate with cuModuleGetFunction)
// ---------------------------------------------------------------------------------

extern "C" {

__global__ __launch_bounds__(1024, 1) void side_aware_memcpy_32(vo0* __restrict__ dst, const vi0* __restrict__ src,
        unsigned int num_elements, unsigned int sm_per_side, __grid_constant__ const param_sm_side_t params) {

    side_aware_elementwise_device<int>(dst, nullptr, src, nullptr, nullptr, nullptr,
                                                num_elements, sm_per_side, params);
}

__global__ __launch_bounds__(1024, 1) void side_aware_memcpy_64(vo0* __restrict__ dst, const vi0* __restrict__ src,
        size_t num_elements, unsigned int sm_per_side, __grid_constant__ const param_sm_side_t params) {

    side_aware_elementwise_device<long long>(dst, nullptr, src, nullptr, nullptr, nullptr,
                                          num_elements, sm_per_side, params);
}

} // extern "C"
