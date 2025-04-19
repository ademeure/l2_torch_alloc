# test.py
import torch
import ctypes
import numpy as np
import time

# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    './alloc.so', 'sideaware_malloc', 'sideaware_free')
# # Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)

# Load the shared library directly for accessing other functions
alloc_lib = ctypes.CDLL('./alloc.so')

# Define the function signatures for the new interface
alloc_lib.get_num_sms.argtypes = []
alloc_lib.get_num_sms.restype = ctypes.c_int
alloc_lib.fill_sm_sides_tensor.argtypes = [ctypes.c_void_p]
alloc_lib.fill_sm_sides_tensor.restype = ctypes.c_bool

# Define all direct accessor functions with the same signature in one loop
for func_name in ['get_num_sm_side0', 'get_num_sm_side1', 'get_min_sm_per_side', 'get_hash_mask']:
    getattr(alloc_lib, func_name).argtypes = []
    getattr(alloc_lib, func_name).restype = ctypes.c_int

# Setup sideaware_memcpy function (this is what gets called as torch.ops.sideaware.memcpy)
alloc_lib.sideaware_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
alloc_lib.sideaware_memcpy.restype = None

# Helper function to get SM side & index as a PyTorch tensor
def get_sm_side_index_tensor():
    # Need to allocate something first to trigger initialization
    dummy = torch.empty((1,), dtype=torch.uint8, device="cuda")

    num_sms = alloc_lib.get_num_sms()
    assert num_sms > 0

    sides_tensor = torch.zeros(num_sms, dtype=torch.uint8, device="cuda")
    tensor_ptr = sides_tensor.data_ptr()
    alloc_lib.fill_sm_sides_tensor(tensor_ptr)
    return sides_tensor

def get_sm_sides_metadata():
    return {
        "num_side0": alloc_lib.get_num_sm_side0(),
        "num_side1": alloc_lib.get_num_sm_side1(),
        "min_sm_per_side": alloc_lib.get_min_sm_per_side(),
        "hash_mask": alloc_lib.get_hash_mask()
    }

# Define the PyTorch operator for manual memcpy
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
    size_in_bytes = src.numel() * src.element_size()

    # Get the current CUDA stream
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device)

    # Call our C function
    alloc_lib.sideaware_memcpy(dst_ptr, src_ptr, size_in_bytes, device, stream)

# Manually define the PyTorch operator
#
sideaware_lib = torch.library.Library("sideaware", "FRAGMENT")

def direct_register_custom_op(op_name, op_func, mutates_args):
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    sideaware_lib.define(op_name + schema_str)
    sideaware_lib.impl(op_name, op_func, "CUDA")

direct_register_custom_op("memcpy", sideaware_memcpy, mutates_args=(["dst"]))

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

# Test with torch.compile
@torch.compile(fullgraph=True)
def compiled_memcpy(dst, src):
    torch.ops.sideaware.memcpy(dst, src)
    return dst  # We return dst here for convenience in the test function

# Create new tensors for the compiled test
src2 = torch.ones((5, 5), dtype=torch.float32, device="cuda") * 2.0
dst2 = torch.zeros((5, 5), dtype=torch.float32, device="cuda")
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

def test_memcpy_correctness(shape, dtype=torch.float32, run_benchmark=False):
    """Test correctness of sideaware_memcpy with tensors of given shape and dtype."""
    print(f"\nTesting shape={shape}, dtype={dtype}")

    # Calculate size in MB
    element_size = torch.empty(1, dtype=dtype).element_size()
    size_mb = np.prod(shape) * element_size / (1024 * 1024)
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
        for i in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            torch.ops.sideaware.memcpy(dst_manual, src)
            end.record()
            torch.cuda.synchronize()
            manual_time = start.elapsed_time(end)
            print(f"\tManual memcpy time: {manual_time:.3f} ms")

            # Copy using PyTorch's built-in
            start.record()
            dst_builtin.copy_(src)
            end.record()
            torch.cuda.synchronize()
            builtin_time = start.elapsed_time(end)
            print(f"\tPyTorch copy_ time: {builtin_time:.3f} ms")
            print(f"\tRatio (manual/builtin): {manual_time/builtin_time:.2f}x")
    else:
        # Just do the copies without timing
        torch.ops.sideaware.memcpy(dst_manual, src)
        dst_builtin.copy_(src)

    # Check correctness
    is_equal = torch.all(dst_manual == dst_builtin).item()
    print(f"Tensors equal: {is_equal}")

    if not is_equal:
        # Show where differences occur
        diff_indices = (dst_manual != dst_builtin).nonzero()
        sample_idx = diff_indices[0].tolist()
        print(f"First difference at index {sample_idx}")
        print(f"!!! Manual value: {dst_manual[tuple(sample_idx)]}")
        print(f"!!! Built-in value: {dst_builtin[tuple(sample_idx)]}")

    return is_equal

# Test with increasing sizes
test_memcpy_correctness((2, 7777))  # Small: <64KiB (not multiple of 16 bytes!)
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
    print(f"Skipped largest test due to: {e}")

print("\n===== MANUAL MEMCPY TESTING COMPLETE =====")
