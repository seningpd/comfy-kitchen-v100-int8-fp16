# comfy-kitchen-v100-int8-fp16

FP16 tensor-core fallback for **INT8 ConvRot** linear layers on NVIDIA **Volta (sm_70)** GPUs, e.g. Tesla V100.

## Why

`comfy_kitchen`'s `int8_linear` has no INT8 tensor-core path on Volta:

| backend | requirement | sm_70 |
|---|---|---|
| CUDA `int8_linear` | `min_compute_capability=(7, 5)` + `_CUBLASLT_AVAILABLE` | never registers |
| Triton `int8_linear` | `min_compute_capability=(8, 0)` | never registers |
| eager fallback | `torch._int_mm` (DP4A-emulated INT8 GEMM) | **runs, and is slower than FP16** |

Measured on a Tesla V100-SXM2-16GB: a 4096鲁 INT8 GEMM via DP4A takes **2.909 ms**, while the same FP16 matmul on the FP16 tensor cores takes **1.507 ms**. So on Volta the INT8 weights cost memory and buy nothing in speed 鈥?and the fallback is **silent**, with no warning that a slower path was selected.

## The key observation

For a ConvRot weight the stored matrix is `W_rot = W @ H岬€` and the activation is rotated online (`x_rot = x @ H`), so the GEMM already runs entirely in the rotated basis:

```
y = x @ W岬€ = (x @ H岬€) @ (W @ H)岬€ = x_rot @ W_rot岬€
```

The dequantized weight therefore must **not** be rotated back. Dequantizing `W_rot` and multiplying by `x_rot` is mathematically equivalent to the original basis, and skipping the inverse Hadamard saves one transform per layer.

This patch dequantizes the INT8 weights to FP16 on the fly (exactly like the existing FP8 path does) and runs the GEMM on the FP16 tensor cores. Weight memory is unchanged.

## Measured results

Real ComfyUI model load (`rayQwen21_turboV13Int8convrot.safetensors`, Qwen-Image-2.1), 225 quantized layers, M=4096 tokens, Tesla V100-SXM2-16GB, torch 2.12.0+cu126, CUDA 12.8:

| | INT8 DP4A (upstream) | this patch |
|---|---|---|
| 225 layers, one pass | 1667.4 ms | **1134.0 ms** (1.47脳) |
| weight memory | 6.96 GB | 6.96 GB (unchanged) |
| peak memory | 8.19 GB | **7.43 GB** |

Accuracy against a float32 reference (same layer, same weights):

| | rel. error vs float32 |
|---|---|
| INT8 DP4A path | 0.00885 |
| this patch | 0.00886 |

Both paths sit on the same INT8 quantization grid, so the distance to the float32 reference is unchanged 鈥?only the GEMM precision differs (int32 accumulate vs FP16 accumulate).

## Contents

| path | what |
|---|---|
| `patches/v100-int8-fp16-fallback.patch` | the upstream patch (254 insertions, 0 deletions) |
| `patches/apply-patch.sh` | helper that applies the patch to a comfy-kitchen checkout |
| `comfyui-plugin/` | drop-in ComfyUI custom node that applies the same fallback at runtime, no source patching required |
| `docs/PR_DESCRIPTION.md` | upstream pull request text |
| `docs/ISSUE_DESCRIPTION.md` | upstream issue text |

## Two ways to use it

**A. ComfyUI plugin (no source patching, recommended for end users)**

Copy `comfyui-plugin/` into `ComfyUI/custom_nodes/`. It replaces the eager backend's `int8_linear` at import time, so it works on any stock install and is disabled automatically on GPUs that do have INT8 tensor cores.

**B. Patch comfy-kitchen itself (for upstream / package users)**

```bash
bash patches/apply-patch.sh /path/to/comfy-kitchen
```

This adds `int8_linear_fp16()` to the eager backend, registers it as a capability, exposes `comfy_kitchen.int8_linear_fp16()`, and makes `int8_linear()` delegate to it on CUDA devices without INT8 tensor cores. Devices with INT8 tensor cores keep the existing path untouched.

## Tests

`tests/test_int8_fp16_fallback.py` 鈥?10 tests covering numerics vs the INT8 path, numerics vs a float32 reference, bias/residual, scalar weight scale, shape validation, capability registration, and the delegation switch.

```
tests/test_int8_fp16_fallback.py: 10 passed
tests/test_int8.py, test_int8_input_act.py, test_int8_residual.py, test_int8_attention.py:
    136 passed, 1154 skipped  (skips = compiled CUDA extension unavailable)
```

## License

The patch is offered under the same license as comfy-kitchen (see `LICENSE` in the upstream repository).
