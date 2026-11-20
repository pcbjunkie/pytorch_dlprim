# YOLOv8n on Radeon OpenCL

Start with the [GCN build and fixture setup](radeon-yolo.md).

| GPU | Input | Training images/s (batch) | Detection FPS (batch) |
| --- | --- | ---: | ---: |
| R9 270 | 256×256 | 35.45 (B48) | 211.39 (B16) |
| R9 270 | 512×512 | 12.64 (B8) | 72.97 (B8) |
| R9 270 | 1024×1024 | 2.99 (B2) | 21.12 (B4) |
| R9 270 | 1920×1080 | 2.12 (B1) | 12.32 (B4) |
| HD 7970 | 256×256 | 49.25 (B64) | 365.39 (B128) |
| HD 7970 | 512×512 | 14.15 (B16) | 116.13 (B32) |
| HD 7970 | 1024×1024 | 3.34 (B4) | 29.94 (B8) |
| HD 7970 | 1920×1080 | 2.31 (B2) | 17.89 (B8) |
| S7150 ×1 | 256×256 | 92.35 (B128) | — |
| S7150 ×1 | 512×512 | 27.38 (B40) | — |
| S7150 ×2 | 512×512 | 48.10 (B16) | 252.28 (B16) |
| S7150 ×2 | 1024×1024 | 12.60 (B8) | 67.92 (B12) |
| S7150 ×2 | 1920×1080 | 10.02 (B6) | 40.56 (B8) |

S7150 ×2 rates aggregate two physical GPUs. Batches are per worker. B4 × 3 means three batch-4 workers per GPU.
These are fixture throughput measurements. See the [README sweep links](../README.md#what-works-today) for other tested batches.

## Opt-ins

- Training uses the [common GCN flags](presets/training-common.env).
- The R9 270 uses its [portable training flags](presets/training-yolov8n-r9-270-portable.env) at 256×256. The 35.45 images/s result uses this preset.
- The training harness enables host prefetch, fused EMA, and the OpenCL TAL and DFL loss adapters.
- Detection uses batched input packing and prefetch, Conv + SiLU, fused SPPF, GPU candidate compaction, and CPU class-aware NMS.
- For HD 5870, HD 6850, and HD 6970, use the [Clover training and detection presets](RADEON_MINIMAL_BASELINE.md).

The [launcher](../tools/run-radeon-yolo.sh) applies these settings. For example, a single S7150 worker at 512×512 uses the following command.

```bash
tools/run-radeon-yolo.sh s7150 yolov8n detect 512 16 results/yolov8n-512.json
```

## Optimizations and fixes

- Coalesced copies, direct concatenation, small-K convolution, shape-tuned GEMM tiles, compact Winograd, and width-first depthwise convolution.
- Inference Conv/BN folding, Conv + SiLU, fused SPPF, and device-side detection candidate filtering.
- Fused SiLU backward, split-K GEMM weight gradients, compact Winograd weight gradients, GEMM/Winograd reuse for input gradients, coalesced BN and depthwise reductions, and chunked SGD/EMA tensor updates.
- Fused TAL metrics and batched top-k scatter, fused CIoU + DFL-16, selective queue retirement, and host batch prefetch.
- Portable local-memory reductions corrected wrong bias and depthwise gradients on Tonga. B192 at 256×256 still needs sustained qualification. Saved mismatches were traced to near ties in SPPF and passed gradient tolerance when pooling winners were aligned.

Shared platform fixes include Tonga grid splitting for large dispatches, the CPDMA fallback for large clears and copies, and portable reductions. See the [Radeon baseline](RADEON_MINIMAL_BASELINE.md) for runtime background.
