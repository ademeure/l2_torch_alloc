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
alloc_lib.fill_sm_sides_tensor.argtypes = [ctypes.c_void_p]
alloc_lib.fill_sm_sides_tensor.restype = ctypes.c_bool

# Returns (num_sms, num_side0, num_side1, min_sm_per_side, hash_mask)
alloc_lib.get_sm_side_summary_ptr.argtypes = []
alloc_lib.get_sm_side_summary_ptr.restype = ctypes.POINTER(ctypes.c_int * 5)
def get_sm_side_summary():
    return tuple(alloc_lib.get_sm_side_summary_ptr().contents)

# Setup sideaware_memcpy function (this is what gets called as torch.ops.sideaware.memcpy)
# void sideaware_memcpy(void* dst, const void* src, size_t size, int device, cudaStream_t stream)
alloc_lib.sideaware_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
alloc_lib.sideaware_memcpy.restype = None

# Expose function to update the custom NVRTC header (triggers recompilation on next call)
# sideaware_set_custom_header returns int header_id
alloc_lib.sideaware_set_custom_header.argtypes = [ctypes.c_char_p]
alloc_lib.sideaware_set_custom_header.restype = ctypes.c_int

# elementwise (generic) API
alloc_lib.sideaware_elementwise.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
alloc_lib.sideaware_elementwise.restype = None

# Helper function to get SM side & index as a PyTorch tensor
def get_sm_side_index_tensor():
    # Need to allocate something first to trigger initialization
    dummy = torch.empty((1,), dtype=torch.uint8, device="cuda")

    num_sms, _, _, _, _ = get_sm_side_summary()
    assert num_sms > 0

    sides_tensor = torch.zeros(num_sms, dtype=torch.uint8, device="cuda")
    tensor_ptr = sides_tensor.data_ptr()
    alloc_lib.fill_sm_sides_tensor(tensor_ptr)
    return sides_tensor

def get_sm_sides_metadata():
    num_sms, num_side0, num_side1, min_side, hash_mask = get_sm_side_summary()
    return {
        "num_sms": num_sms,
        "num_side0": num_side0,
        "num_side1": num_side1,
        "min_sm_per_side": min_side,
        "hash_mask": hash_mask,
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

    # Make sure src and dst are contiguous
    assert src.is_contiguous(), "Source tensor must be contiguous"
    assert dst.is_contiguous(), "Destination tensor must be contiguous"

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

# ---------------------------------------------------------------------------
# Demonstrate dynamic header injection & recompilation via NVRTC
# ---------------------------------------------------------------------------
print("\nTesting NVRTC dynamic recompilation path (inject custom header)...")

# Prepare a minimal header that adds an unused device helper.  In a real use
# case this could define e.g. a GELU or ReLU transformation that will be used
# by an element‑wise kernel variant compiled on‑the‑fly.
header_code = b"__device__ float dummy_scale(float x) { return x * 1.0f; }\n"

# Tell the allocator to use the new header for future kernel compilations.
# The next sideaware_memcpy call will trigger recompilation automatically.
header_id = alloc_lib.sideaware_set_custom_header(header_code)

src_hdr = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
dst_hdr = torch.zeros_like(src_hdr)

torch.ops.sideaware.memcpy(dst_hdr, src_hdr)

# Verify correctness after recompilation
assert torch.all(dst_hdr == src_hdr)
print("Header recompilation successful – memcpy still correct.")

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

def test_memcpy_correctness(shape, dtype=torch.float32, run_benchmark=False, start_idx=0):
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
test_memcpy_correctness((2, 7777), start_idx=1)  # Small: <64KiB (not multiple of 16 bytes!)
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
