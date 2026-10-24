## Summary

On NVIDIA Volta (sm_70, e.g. V100) `int8_linear` is **slower** than an FP16 GEMM, because Volta has no INT8 tensor core:

- The CUDA backend's `int8_linear` requires `min_compute_capability=(7, 5)` **and** `_CUBLASLT_AVAILABLE`, so it never registers on sm_70.
- The Triton backend's `int8_linear` requires `min_compute_capability=(8, 0)`.
- What actually runs is the eager fallback, which uses `torch._int_mm` — a **DP4A-emulated** INT8 GEMM. Measured on a V100: 4096³ in **2.909 ms**, versus **1.507 ms** for the same FP16 matmul on the FP16 tensor cores.

So the INT8 weights cost memory and buy nothing in speed, and the fallback is silent — there is no warning that a 2x-slower path was selected.

This PR adds an FP16-tensor-core fallback for devices without INT8 tensor cores. The INT8 weights stay in memory (dequantized to FP16 on the fly, exactly like the existing FP8 path), so the memory footprint is unchanged.

## The key observation

For a ConvRot weight the stored matrix is `W_rot = W @ H^T` and the activation is rotated online (`x_rot = x @ H`), so **the GEMM runs entirely in the rotated basis**:

```
y = x @ W^T = (x @ H^T) @ (W @ H)^T = x_rot @ W_rot^T
```

That means the dequantized weight must **not** be rotated back. Dequantizing `W_rot` and multiplying by `x_rot` is mathematically equivalent to the original basis, and skipping the inverse Hadamard saves one transform per layer.

The activation is still quantized to INT8 with the same grid as the INT8 path, so the numerics stay on the INT8 quantization grid and only the GEMM changes.

## Measured results

Real ComfyUI model load (`rayQwen21_turboV13Int8convrot.safetensors`, Qwen-Image-2.1), 225 quantized layers, M=4096 tokens, Tesla V100-SXM2-16GB:

| | INT8 DP4A (current) | this PR |
|---|---|---|
| 225 layers, one pass | 1667.4 ms | **1134.0 ms** (**1.47x**) |
| weight memory | 6.96 GB | 6.96 GB (unchanged) |
| peak memory | 8.19 GB | **7.43 GB** |

Accuracy against a float32 reference (same layer, same weights):

| | rel. error vs float32 |
|---|---|
| INT8 DP4A path | 0.00885 |
| this PR | 0.00886 |

The two paths sit on the same quantization grid, so the distance to the float32 reference is unchanged — only the GEMM precision differs (int32 accumulate vs FP16 accumulate).

## Changes

- `comfy_kitchen/backends/eager/quantization.py`
  - new `int8_linear_fp16()`: INT8 linear that dequantizes to FP16 on the fly and runs the GEMM on the FP16 tensor cores
  - new `_cuda_device_has_int8_tensor_core()` helper, cached per device (mirrors the existing `_cuda_device_is_turing()` pattern)
  - `int8_linear()` now delegates to `int8_linear_fp16()` on CUDA devices without INT8 tensor cores. Devices with INT8 tensor cores keep the existing path untouched.
- `comfy_kitchen/backends/eager/__init__.py`: register `int8_linear_fp16` as an eager capability
- `comfy_kitchen/__init__.py`: public `comfy_kitchen.int8_linear_fp16()` entry point, plus a `comfy_kitchen::int8_linear_fp16` custom op for `torch.compile` compatibility
- `tests/test_int8_fp16_fallback.py`: 10 tests covering numerics vs the INT8 path, numerics vs a float32 reference, bias/residual, scalar weight scale, shape validation, capability registration, and the delegation switch

Purely additive (254 insertions, 0 deletions).

## Notes for the reviewer

- The activation is deliberately still quantized to INT8 in this PR, to keep `int8_linear`'s numerical contract intact. A variant that keeps the activation in FP16 is ~25% faster again (225 layers: 1134 ms -> 859 ms, error vs float32 0.0017), but it changes the quantization grid, so it seemed better kept out of this PR. Happy to split it out if that is preferred.
- `quantize_int8_rowwise` in the CUDA backend requires sm_80+ and raises `no kernel image is available for execution on the device` on sm_70 rather than falling back gracefully. That is a separate issue (reported separately), but it is worth a graceful fallback there too.
- The eager fallback currently selects the DP4A path with no diagnostic. A one-time warning when a device without INT8 tensor cores takes the eager path would have made this much easier to notice.

## Test plan

- `tests/test_int8_fp16_fallback.py`: 10 passed
- `tests/test_int8.py`, `tests/test_int8_input_act.py`, `tests/test_int8_residual.py`, `tests/test_int8_attention.py`: 136 passed, 1154 skipped (skips are the compiled CUDA extension, unavailable in this environment)

Hardware used for the measurements: Tesla V100-SXM2-16GB, torch 2.12.0+cu126, CUDA 12.8.
