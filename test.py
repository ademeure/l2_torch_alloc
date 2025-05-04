# test.py
import torch
import ctypes
import numpy as np
import time

# Use the sideaware allocator
sideaware_alloc = torch.cuda.memory.CUDAPluggableAllocator('./sideaware.so', 'sideaware_malloc_auto', 'sideaware_free_auto')
torch.cuda.memory.change_current_allocator(sideaware_alloc)

# Load the allocator's library directly to access other functions
lib = ctypes.CDLL('./sideaware.so')

# Sideaware memcpy (i.e. "default kernel" when no custom kernel is provided via sideaware_compile)
lib.sideaware_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
lib.sideaware_memcpy.restype = None

# Compile custom elementwise function (returns kernel_id which can be passed to sideaware_elementwise)
lib.sideaware_compile.argtypes = [ctypes.c_char_p, ctypes.c_bool]
lib.sideaware_compile.restype = ctypes.c_int

# Sideaware single-input / single-output elementwise API (simple version)
lib.sideaware_one_to_one.argtypes = [ctypes.c_int, ctypes.c_size_t, # kernel_id, num_bytes
                                           ctypes.c_void_p, ctypes.c_void_p, # out0, in0
                                           ctypes.c_int, ctypes.c_void_p] # device, stream
lib.sideaware_one_to_one.restype = None

# Sideaware multi-input / multi-output elementwise API (advanced version)
lib.sideaware_elementwise.argtypes = [ctypes.c_int, ctypes.c_size_t, # kernel_id, num_bytes
                                      ctypes.c_void_p, ctypes.c_void_p, # out0, out1
                                      ctypes.c_void_p, ctypes.c_void_p, # out2, out3
                                      ctypes.c_void_p, ctypes.c_void_p, # in0, in1
                                      ctypes.c_void_p, ctypes.c_void_p, # in2, in3
                                      ctypes.c_void_p, ctypes.c_void_p, # sideband_ptr, sideband_value
                                      ctypes.c_int, ctypes.c_int, # parallel_chunks, forced_sm_per_side
                                      ctypes.c_int, ctypes.c_void_p] # device, stream
lib.sideaware_elementwise.restype = None

# Useful for custom kernels (e.g. matmul with sideaware B matrix) and debug logs
lib.fill_sm_sides_tensor.argtypes = [ctypes.c_void_p]
lib.fill_sm_sides_tensor.restype = ctypes.c_bool

# Returns (num_sms, num_side0, num_side1, min_sm_per_side, hash_mask)
lib.get_sm_side_summary_ptr.argtypes = []
lib.get_sm_side_summary_ptr.restype = ctypes.POINTER(ctypes.c_int * 5)
def get_sm_side_summary():
    return tuple(lib.get_sm_side_summary_ptr().contents)

# Helper function to get SM side & index as a PyTorch tensor
def get_sm_side_index_tensor():
    num_sms, _, _, _, _ = get_sm_side_summary()
    sides_tensor = torch.zeros(num_sms, dtype=torch.uint8, device="cuda")
    tensor_ptr = sides_tensor.data_ptr()
    lib.fill_sm_sides_tensor(tensor_ptr)
    return sides_tensor

# Returns dictionary version of get_sm_side_summary()
def get_sm_sides_metadata():
    num_sms, num_side0, num_side1, min_side, hash = get_sm_side_summary()
    return { "num_sms": num_sms, "num_side0": num_side0, "num_side1": num_side1, "min_side": min_side, "hash": hash }

# Define the PyTorch operators for sideaware memcpy, sideaware one_to_one, and sideaware elementwise
def sideaware_memcpy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """
    Performs a manual memcpy from src tensor to dst tensor.
    Modifies dst in-place.

    Args:
        dst: Destination tensor (must be on CUDA)
        src: Source tensor (must be on CUDA)
    """
    # Validate inputs
    assert dst.device.type == "cuda" and src.device.type == "cuda", "Both tensors must be on CUDA"
    assert dst.dtype == src.dtype, "Source and destination must have the same dtype"
    assert dst.numel() >= src.numel(), "Destination tensor must be at least as large as source"

    # Get pointers and size
    dst_ptr = dst.data_ptr()
    src_ptr = src.data_ptr()
    num_bytes = src.numel() * src.element_size()

    # Make sure src and dst are contiguous and aligned
    assert src.is_contiguous(), "Source tensor must be contiguous"
    assert dst.is_contiguous(), "Destination tensor must be contiguous"
    assert (dst_ptr % 16 == 0) and (src_ptr % 16 == 0), "Destination and source must be 16-byte aligned"

    # Call our C function
    lib.sideaware_memcpy(dst_ptr, src_ptr, num_bytes, torch.cuda.current_device(), torch.cuda.current_stream())

def sideaware_one_to_one(kernel_id: int, dst: torch.Tensor, src: torch.Tensor) -> None:
    assert dst is not None or src is not None
    src_bytes = dst is not None and dst.numel() * dst.element_size() or 0
    dst_bytes = dst is not None and dst.numel() * dst.element_size() or 0
    num_bytes = max(src_bytes, dst_bytes)

    # Make sure src and dst are contiguous and aligned
    assert src is None or (src.is_contiguous() and src.device.type == "cuda"), "Source tensor 0 must be contiguous"
    assert dst is None or (dst.is_contiguous() and dst.device.type == "cuda"), "Destination tensor 0 must be contiguous"

    # Get pointers and size
    dst_ptr = dst is not None and dst.data_ptr() or 0
    src_ptr = src is not None and src.data_ptr() or 0
    assert dst_ptr % 16 == 0 and src_ptr % 16 == 0, "Destination and source tensors must be 16-byte aligned"

    # Call our C function
    lib.sideaware_one_to_one(kernel_id, num_bytes, dst_ptr, src_ptr,
                             torch.cuda.current_device(), torch.cuda.current_stream())

def sideaware_elementwise(kernel_id: int,
                          out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor, out3: torch.Tensor,
                          in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, in3: torch.Tensor,
                          sideband_tensor: torch.Tensor = None, sideband_value: int = 0,
                          parallel_chunks: int = 0, forced_sm_per_side: int = 0) -> None:
    assert out0 is not None or in0 is not None
    src_bytes = out0 is not None and out0.numel() * out0.element_size() or 0
    dst_bytes = out0 is not None and out0.numel() * out0.element_size() or 0
    num_bytes = max(src_bytes, dst_bytes)

    # Make sure src and dst are contiguous and aligned
    assert in0 is None or (in0.is_contiguous() and in0.device.type == "cuda"), "Source tensor 0 must be contiguous"
    assert in1 is None or (in1.is_contiguous() and in1.device.type == "cuda"), "Source tensor 1 must be contiguous"
    assert in2 is None or (in2.is_contiguous() and in2.device.type == "cuda"), "Source tensor 2 must be contiguous"
    assert in3 is None or (in3.is_contiguous() and in3.device.type == "cuda"), "Source tensor 3 must be contiguous"
    assert out0 is None or (out0.is_contiguous() and out0.device.type == "cuda"), "Destination tensor 0 must be contiguous"
    assert out1 is None or (out1.is_contiguous() and out1.device.type == "cuda"), "Destination tensor 1 must be contiguous"
    assert out2 is None or (out2.is_contiguous() and out2.device.type == "cuda"), "Destination tensor 2 must be contiguous"
    assert out3 is None or (out3.is_contiguous() and out3.device.type == "cuda"), "Destination tensor 3 must be contiguous"

    # Get pointers and size
    out0_ptr = out0 is not None and out0.data_ptr() or 0
    out1_ptr = out1 is not None and out1.data_ptr() or 0
    out2_ptr = out2 is not None and out2.data_ptr() or 0
    out3_ptr = out3 is not None and out3.data_ptr() or 0
    in0_ptr = in0 is not None and in0.data_ptr() or 0
    in1_ptr = in1 is not None and in1.data_ptr() or 0
    in2_ptr = in2 is not None and in2.data_ptr() or 0
    in3_ptr = in3 is not None and in3.data_ptr() or 0
    sideband_ptr = sideband_tensor is not None and sideband_tensor.data_ptr() or 0

    assert out0_ptr % 16 == 0 and out1_ptr % 16 == 0 and out2_ptr % 16 == 0 and out3_ptr % 16 == 0, "Destination tensors must be 16-byte aligned"
    assert in0_ptr % 16 == 0 and in1_ptr % 16 == 0 and in2_ptr % 16 == 0 and in3_ptr % 16 == 0, "Source tensors must be 16-byte aligned"

    # Call our C function
    lib.sideaware_elementwise(kernel_id, num_bytes,
                                    out0_ptr, out1_ptr, out2_ptr, out3_ptr,
                                    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
                                    sideband_ptr, sideband_value, parallel_chunks, forced_sm_per_side,
                                    torch.cuda.current_device(), torch.cuda.current_stream())

# Manually define the PyTorch operator
def direct_register_custom_op(op_lib, op_name, op_func, mutates_args):
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    op_lib.define(op_name + schema_str)
    op_lib.impl(op_name, op_func, "CUDA")

sideaware_lib = torch.library.Library("sideaware", "FRAGMENT")
direct_register_custom_op(sideaware_lib, "memcpy", sideaware_memcpy, mutates_args=(["dst"]))
direct_register_custom_op(sideaware_lib, "one_to_one", sideaware_one_to_one, mutates_args=(["dst"]))
direct_register_custom_op(sideaware_lib, "elementwise", sideaware_elementwise, mutates_args=(["out0", "out1", "out2", "out3", "sideband_tensor"]))

# Let's run the new functions
print("Running SM side detection...")

# Get the SM sides tensor
sides_tensor = get_sm_side_index_tensor()
print(f"SM sides tensor: {sides_tensor}")

# Get the metadata using the direct accessor functions
metadata = get_sm_sides_metadata()
print(f"SM sides metadata: {metadata}")

# Test our manual memcpy function
print("\nTesting manual memcpy:")
src_tensor = torch.ones((10, 10), dtype=torch.float32, device="cuda")
dst_tensor = torch.zeros((10, 10), dtype=torch.float32, device="cuda")
print(f"Before memcpy - src: {src_tensor[0, 0]} & {src_tensor[9, 9]}, dst: {dst_tensor[0, 0]} & {dst_tensor[9, 9]}")

# Call our custom op
torch.ops.sideaware.memcpy(dst_tensor, src_tensor)
print(f"After memcpy - src: {src_tensor[0, 0]} & {src_tensor[9, 9]}, dst: {dst_tensor[0, 0]} & {dst_tensor[9, 9]}")

# ---------------------------------------------------------------------------
# Demonstrate dynamic header injection & recompilation via NVRTC
# ---------------------------------------------------------------------------
print("\nTesting NVRTC dynamic recompilation path (custom sideaware kernel)...")

header_code = b"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

typedef float o0;
typedef half o1;
typedef __nv_bfloat16 i0;
typedef float i1;

struct unused {};
typedef unused o2, o3, i2, i3;

template<typename size_type>
__device__ __forceinline__ void elementwise_op(
        size_type element_idx, o0 &out0, o1 &out1, o2 &out2, o3 &out3,
        const i0 &in0, const i1 &in1, const i2 &in2, const i3 &in3)
{
    out0 = (o0)((float)in0 + (float)in1);
    out1 = (o1)((float)in1 * 10.0f);
}

constexpr int unrolled = 4; // unrolled loop iterations (increases register pressure especially with multiple inputs)
constexpr bool reverse_order = true; // process end of array 1st (maximise L2 hits with normal->reverse->normal->...)
constexpr bool input_evict[4] = {true, true, true, true}; // do not keep inputs in L2 (more space for outputs)

constexpr bool support_concurrent_kernels = false; // use atomics to dynamically assign SM side index
constexpr bool input_discard[4] = {0}; // danger: discards from L2 *before* data is written to DRAM
"""

in0_hdr = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
in1_hdr = torch.ones_like(in0_hdr) * 2
out0_hdr = torch.zeros_like(in1_hdr)
out1_hdr = torch.zeros_like(in1_hdr)

in0_hdr = in0_hdr.to(torch.bfloat16)
out1_hdr = out1_hdr.to(torch.float16)

kernel_id = lib.sideaware_compile(header_code, True)
torch.ops.sideaware.elementwise(kernel_id, out0_hdr, out1_hdr, None, None, in0_hdr, in1_hdr, None, None, None, 0, 0, 0)

# Verify correctness after recompilation
if not torch.all(out0_hdr == (in0_hdr+in1_hdr)) or not torch.all(out1_hdr == (in1_hdr*10.0)):
    print("!!!!! Incorrect output for custom elementwise kernel !!!!!")
    print(f"in0_hdr: {in0_hdr}")
    print(f"in1_hdr: {in1_hdr}")
    print(f"out0_hdr: {out0_hdr}")
    print(f"out1_hdr: {out1_hdr}")
else:
    print("Successfully compiled and executed the custom elementwise kernel")

# Test with torch.compile
@torch.compile(fullgraph=True)
def compiled_memcpy(dst, src):
    torch.ops.sideaware.memcpy(dst, src)
    return dst  # We return dst here for convenience in the test function

# Create new tensors for the compiled test
src2 = torch.ones((8, 8), dtype=torch.float32, device="cuda") * 2.0
dst2 = torch.zeros((8, 8), dtype=torch.float32, device="cuda")
print(f"Before compiled memcpy - src: {src2[0, 0]} & {src2[4, 4]}, dst: {dst2[0, 0]} & {dst2[4, 4]}")

# Call compiled function
result2 = compiled_memcpy(dst2, src2)
print(f"After compiled memcpy - src: {src2[0, 0]} & {src2[4, 4]}, dst: {dst2[0, 0]} & {dst2[4, 4]}")

# Original test code
print("\nRunning original allocation tests:")
for factor in (1024, 1024 ** 2):
        print(f"Allocate 2 * {20 * factor} bytes of memory on the GPU from Python")
        data = torch.empty((20, factor), dtype=torch.uint8, device="cuda")
        data2 = torch.empty((20, factor), dtype=torch.uint8, device="cuda")
        print(f"Free 2 * {20 * factor} bytes of memory on the GPU from Python")
        del data
        del data2
        print("Python side: memory is released")

        print(f"Allocate {30 * factor} bytes of memory on the GPU from Python")
        data = torch.empty((30, factor), dtype=torch.uint8, device="cuda")
        print(f"Free {30 * factor} bytes of memory on the GPU from Python")
        del data
        print("Python side: memory is released")

        print(f"Allocate {50 * factor} bytes of memory on the GPU from Python")
        data = torch.empty((50, factor), dtype=torch.uint8, device="cuda")
        print(f"Free {50 * factor} bytes of memory on the GPU from Python")
        del data
        print("Python side: memory is released")

# Comprehensive testing for sideaware_memcpy
print("\n===== COMPREHENSIVE MANUAL MEMCPY TESTING =====")

def test_memcpy_correctness(shape, dtype=torch.float32, run_benchmark=False, start_idx=0):
    """Test correctness of sideaware_memcpy with tensors of given shape and dtype."""
    print(f"\nTesting shape={shape}, dtype={dtype}")

    # Calculate size in MB (not MiB, because bandwidth is typically GB/s rather than GiB/s)
    element_size = torch.empty(1, dtype=dtype).element_size()
    size_mb = np.prod(shape) * element_size / (1000 * 1000)
    print(f"Tensor size: {size_mb:.2f} MB")

    # Create source tensor with recognizable pattern
    src = torch.arange(np.prod(shape), dtype=dtype, device="cuda").reshape(shape)

    # Create destination tensors for our method and PyTorch's built-in
    dst_manual = torch.zeros_like(src)
    dst_builtin = torch.zeros_like(src)

    # Warm-up CUDA
    torch.cuda.synchronize()

    # Copy using our manual memcpy
    if run_benchmark:
        for i in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            torch.ops.sideaware.memcpy(dst_manual[start_idx:], src[start_idx:])
            end.record()
            torch.cuda.synchronize()
            manual_time = start.elapsed_time(end)
            print(f"\tManual memcpy time: {manual_time:.3f} ms")

            # Copy using PyTorch's built-in
            start.record()
            dst_builtin[start_idx:].copy_(src[start_idx:])
            end.record()
            torch.cuda.synchronize()
            builtin_time = start.elapsed_time(end)
            print(f"\tPyTorch copy_ time: {builtin_time:.3f} ms")
            print(f"\tRatio (manual/builtin): {manual_time/builtin_time:.2f}x")
    else:
        # Just do the copies without timing
        torch.ops.sideaware.memcpy(dst_manual[start_idx:], src[start_idx:])
        dst_builtin[start_idx:].copy_(src[start_idx:])

    # Check correctness
    is_equal = torch.all(dst_manual == dst_builtin).item()
    print(f"Tensors equal: {is_equal}")

    if not is_equal:
        # Show up to 5 differences
        diff_indices = (dst_manual != dst_builtin).nonzero()
        next_diffs = diff_indices[0:5]
        for idx in next_diffs:
            print(f"--- Index: {idx}")
            print(f"!!! Output: {dst_manual[tuple(idx)]}")
            print(f"!!! Expected: {dst_builtin[tuple(idx)]}")


    return is_equal

# Test with increasing sizes
test_memcpy_correctness((4, 777))  # Tiny
test_memcpy_correctness((100000, 15000))  # Huge: 6 GB

# Test different dtypes
test_memcpy_correctness((1000, 1000), dtype=torch.int16)
test_memcpy_correctness((1000, 1000), dtype=torch.float64)

# Test with a very large tensor and run benchmark
test_memcpy_correctness((10000, 1000), dtype=torch.float32, run_benchmark=True)  # ~40 MB

# Test with an even larger tensor if enough GPU memory
try:
    test_memcpy_correctness((20000, 20000), dtype=torch.float32, run_benchmark=True)  # ~1600 MB
except RuntimeError as e:
    print(f"Skipped large test due to: {e}")

try:
    test_memcpy_correctness((200000, 20000), dtype=torch.float32, run_benchmark=True)  # ~16000 MB
except RuntimeError as e:
    print(f"Skipped large test due to: {e}")


print("\n===== MANUAL MEMCPY TESTING COMPLETE =====")
