## Environment

- GPU: NVIDIA Tesla V100-SXM2-16GB (compute capability 7.0, Volta)
- comfy-kitchen: 0.2.40 (`f8b7e97`)
- ComfyUI: 0.42.0
- torch: 2.12.0+cu126, CUDA 12.8
- Model: `rayQwen21_turboV13Int8convrot.safetensors` (Qwen-Image-2.1, ConvRot int8)

## What happens

On sm_70 there is no INT8 tensor core, so `int8_linear` ends up in the eager fallback, which uses `torch._int_mm` (cuBLASLt DP4A emulation). That path is *slower* than a plain FP16 matmul on the same GPU:

| operation (4096³) | time |
|---|---|
| `torch._int_mm` (INT8, DP4A) | 2.909 ms |
| `torch.matmul` (FP16 tensor cores) | **1.507 ms** |

Measured end-to-end on the real model (225 quantized layers, M=4096 tokens):

| | time | weight memory | peak memory |
|---|---|---|---|
| INT8 DP4A (current) | 1667.4 ms | 6.96 GB | 8.19 GB |
| FP16 fallback | 1134.0 ms | 6.96 GB | 7.43 GB |

So the INT8 weights buy memory but cost roughly 1.5x in speed, with **no warning emitted**.

## Why it is easy to miss

1. The CUDA backend's `int8_linear` is gated on `min_compute_capability=(7, 5)` **and** `_CUBLASLT_AVAILABLE`, so on sm_70 it simply never registers. The Triton one needs sm_80+. Both exclusions are silent.
2. The eager fallback is the "portable reference" implementation, so falling back to it looks intentional rather than degraded.
3. Nothing in the logs says which path was taken.

For comparison, the Triton FP16 path is also unusable here: `tl.dot` on sm_70 produces correct results but falls back to scalar FMA (930 ms vs 1.43 ms for `torch.matmul`, ~650x slower), because sm_70 does not generate FP16 MMA.

## Two separate problems

### 1. No diagnostic on the slow fallback

There is no way for a user to tell that a 1.5x-slower path was chosen. A one-time warning when a device without INT8 tensor cores takes the eager path would make this visible. (A PR for the FP16 fallback itself is filed separately.)

### 2. `quantize_int8_rowwise` raises instead of falling back

The CUDA backend's `quantize_int8_rowwise` requires sm_80+. On sm_70 it does not fall back to the eager implementation, it raises:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

The eager backend has a perfectly good pure-PyTorch `quantize_int8_rowwise`, so this could degrade gracefully instead of throwing. (`backends/cuda/__init__.py`, `quantize_int8_rowwise` constraint has `default_devices=cuda_devices` but no `min_compute_capability` guard for the kernel itself.)

## Suggested fix

For the fallback itself: dequantize the INT8 weights to FP16 on the fly and run the GEMM on the FP16 tensor cores. The memory footprint stays that of the INT8 model, because the FP16 copy is a temporary (this is what the FP8 path already does).

For a ConvRot weight there is a useful detail: the stored matrix is `W_rot = W @ H^T` and the activation is rotated online (`x_rot = x @ H`), so the GEMM runs entirely in the rotated basis and the dequantized weight must **not** be rotated back. That makes the fallback exact, and saves an inverse Hadamard per layer.

## Question

Is the intended behaviour for pre-Turing NVIDIA GPUs to be "INT8 is memory-only, compute in FP16"? If so, it would be good to make that explicit rather than relying on the user to notice the slowdown.
