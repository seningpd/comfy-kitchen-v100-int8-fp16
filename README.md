# comfy-kitchen-v100-int8-fp16

FP16 tensor-core fallback for **INT8 ConvRot** linear layers on NVIDIA **Volta (sm_70)** GPUs, e.g. Tesla V100.

This is the source-patch form, for people who want to patch `comfy-kitchen` itself or follow the upstream PR.

**Most users want the ComfyUI plugin instead** - it applies the same fix at runtime with no source patching:
[**seningpd/ComfyUI-V100-Precision-Control**](https://github.com/seningpd/ComfyUI-V100-Precision-Control)

## Why

`comfy_kitchen`'s `int8_linear` has no usable INT8 tensor-core path on Volta:

| backend | requirement | sm_70 |
|---|---|---|
| CUDA `int8_linear` | `min_compute_capability=(7, 5)` + `_CUBLASLT_AVAILABLE` | never registers |
| Triton `int8_linear` | `min_compute_capability=(8, 0)` | never registers |
| eager fallback | `torch._int_mm` (DP4A-emulated INT8 GEMM) | **runs, and is slower than FP16** |

Measured on a Tesla V100-SXM2-16GB: a 4096^3 INT8 GEMM via DP4A takes **2.909 ms**, while the same FP16 matmul on the FP16 tensor cores takes **1.507 ms**. So on Volta the INT8 weights cost memory and buy nothing in speed - and the fallback is **silent**, with no warning that a slower path was selected.

## The key observation

For a ConvRot weight the stored matrix is `W_rot = W @ H^T` and the activation is rotated online (`x_rot = x @ H`), so the GEMM already runs entirely in the rotated basis:

```
y = x @ W^T = (x @ H^T) @ (W @ H)^T = x_rot @ W_rot^T
```

The dequantized weight therefore must **not** be rotated back. Dequantizing `W_rot` and multiplying by `x_rot` is mathematically equivalent to the original basis, and skipping the inverse Hadamard saves one transform per layer.

This patch dequantizes the INT8 weights to FP16 on the fly (exactly like the existing FP8 path does) and runs the GEMM on the FP16 tensor cores. Weight memory is unchanged.

## Measured results

Real ComfyUI model load (`rayQwen21_turboV13Int8convrot.safetensors`, Qwen-Image-2.1), 225 quantized layers, M=4096 tokens, Tesla V100-SXM2-16GB, torch 2.12.0+cu126, CUDA 12.8:

| | INT8 DP4A (upstream) | this patch |
|---|---|---|
| 225 layers, one pass | 1667.4 ms | **1134.0 ms** (1.47x) |
| weight memory | 6.96 GB | 6.96 GB (unchanged) |
| peak memory | 8.19 GB | **7.43 GB** |

Accuracy against a float32 reference (same layer, same weights):

| | rel. error vs float32 |
|---|---|
| INT8 DP4A path | 0.00885 |
| this patch | 0.00886 |

Both paths sit on the same INT8 quantization grid, so the distance to the float32 reference is unchanged - only the GEMM precision differs (int32 accumulate vs FP16 accumulate).

### Where the remaining time goes

The patch keeps the activation quantized to INT8 on purpose (to preserve `int8_linear`'s numerical contract), which costs an extra quantize pass per layer.

If the activation is left in FP16 instead - the INT8 grid exists only to feed `torch._int_mm`, which is no longer used - the same 225 layers drop to **859.2 ms (1.94x)** and peak memory to **7.30 GB**. On a real layer set (MiniMax H3, 200 quantized layers, M=4096) the FP16-activation variant lands within 2.2% of an equivalent FP8 model, whereas keeping the activation on the INT8 grid costs about 8%:

| variant | 200 layers | vs FP8 model |
|---|---|---|
| FP8 model (tensorwise scalar scale) | 1884.3 ms | 1.000x |
| INT8 + FP16 activation, no INT8 grid | 1925.9 ms | 1.022x |

That variant changes the numerics relative to the INT8 reference path, so it is deliberately **not** part of this patch - it is what the [ComfyUI plugin](https://github.com/seningpd/ComfyUI-V100-Precision-Control) does.

## Contents

| path | what |
|---|---|
| `patches/v100-int8-fp16-fallback.patch` | the upstream patch (4 files, +737/-7) |
| `patches/apply-patch.sh` | helper that applies the patch to a comfy-kitchen checkout |
| `tests/test_int8_fp16_fallback.py` | 27 tests covering numerics, options, validation, capability detection and dispatch |
| `docs/PR_DESCRIPTION.md` | upstream pull request text |
| `docs/ISSUE_DESCRIPTION.md` | upstream issue text |

## Two ways to use it

**A. ComfyUI plugin (no source patching, recommended for end users)**

See [seningpd/ComfyUI-V100-Precision-Control](https://github.com/seningpd/ComfyUI-V100-Precision-Control) - copy one folder into `ComfyUI/custom_nodes/`. It replaces the eager backend's `int8_linear` at import time, works on any stock install, is a no-op on GPUs that do have INT8 tensor cores, and additionally forces FP16 compute for Qwen-Image-2.1.

**B. Patch comfy-kitchen itself (for upstream / package users)**

```bash
bash patches/apply-patch.sh /path/to/comfy-kitchen
```

This adds `int8_linear_fp16()` to the eager backend, registers it as a capability, exposes `comfy_kitchen.int8_linear_fp16()`, and makes `int8_linear()` delegate to it on CUDA devices without INT8 tensor cores. Devices with INT8 tensor cores keep the existing path untouched.

The patch is generated against comfy-kitchen **v0.2.40 (commit `f8b7e97`)**. On a different revision it may not apply cleanly; `apply-patch.sh` uses `git apply --3way` and will tell you.

## When the fallback kicks in

`int8_linear` delegates to the FP16 fallback only when **all** of these hold:

- the device has no INT8 tensor core (`major >= 8` or `7.x minor >= 5` means it does)
- the device **has** FP16 tensor cores (`major >= 7`) - so Pascal and Maxwell are excluded, where FP16 runs at a fraction of the DP4A INT8 rate
- the activation is not FP32, unless `COMFY_KITCHEN_INT8_FP16_FALLBACK_FP32=1` is set

FP32 is excluded by default because an FP32 activation is the caller explicitly asking for FP32 numerics (a NaN/overflow fuse re-running a block, or a model whose compute dtype is pinned). Downgrading it to FP16 is 2.27x faster (86.64 ms -> 38.30 ms, H3 INT8, 4 layers, M=4096) but trades away 13 mantissa bits, so it is opt-in.

Two escape hatches:

| variable | default | effect |
|---|---|---|
| `COMFY_KITCHEN_INT8_FP16_FALLBACK=0` | `1` | keep the original INT8 path everywhere (A/B testing) |
| `COMFY_KITCHEN_INT8_FP16_FALLBACK_FP32=1` | unset | also route FP32 activations through the FP16 fallback |

Compute dtype is chosen by `_int8_fp16_compute_dtype()`: FP16 below Ampere, the activation's own dtype on Ampere and later. Volta and Turing have no BF16 tensor cores, so a BF16 GEMM there takes a slow conversion path (13.449 ms for 4096^3 versus 1.535 ms for FP16).

## Tests

```
tests/test_int8_fp16_fallback.py                       27 passed
tests/test_int8.py, test_int8_input_act.py,
tests/test_int8_residual.py, test_int8_attention.py    136 passed, 1154 skipped
```

(the skips are the compiled CUDA extension, unavailable on a V100-only box)

## Relationship to upstream

Upstream pull request: [Comfy-Org/comfy-kitchen#277](https://github.com/Comfy-Org/comfy-kitchen/pull/277)
Related issue: [Comfy-Org/comfy-kitchen#278](https://github.com/Comfy-Org/comfy-kitchen/issues/278)

The patch here tracks that PR. Until it is merged, this repository is the patch form and the plugin repository is the zero-install form.

## License

Offered under the same license as comfy-kitchen (see `LICENSE` in the upstream repository).
