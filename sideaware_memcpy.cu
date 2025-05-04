#define KERNEL_NAME side_aware_memcpy

struct unused {};
typedef int o0, i0;
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

#include "sideaware_kernel.cuh"
