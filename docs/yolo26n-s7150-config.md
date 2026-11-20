# YOLO26n on Radeon OpenCL

Start with the [GCN build and fixture setup](radeon-yolo.md).

| GPU | Input | Training images/s (batch) | Detection FPS (batch) |
| --- | --- | ---: | ---: |
| R9 270 | 256×256 | 31.00 (B48) | 240.15 (B16) |
| R9 270 | 512×512 | 8.20 (B8) | 74.98 (B8) |
| R9 270 | 1024×1024 | 1.90 (B2) | 19.94 (B4) |
| R9 270 | 1920×1080 | 1.17 (B1) | 12.14 (B4) |
| HD 7970 | 256×256 | 33.99 (B48) | 508.15 (B128) |
| HD 7970 | 512×512 | 8.79 (B16) | 125.67 (B32) |
| HD 7970 | 1024×1024 | 2.06 (B4) | 30.42 (B8) |
| HD 7970 | 1920×1080 | 0.88 (B1) | 17.51 (B8) |
| S7150 ×2 | 512×512 | 37.16 (B16) | 267.99 (B16) |
| S7150 ×2 | 1024×1024 | 9.75 (B8) | 61.42 (B12) |
| S7150 ×2 | 1920×1080 | 7.98 (B6) | 38.49 (B8) |

S7150 ×2 rates aggregate two physical GPUs. Batches are per worker. B4 × 3 means three batch-4 workers per GPU.
These are fixture throughput measurements. See the [README sweep links](../README.md#what-works-today) for other tested batches.

## Opt-ins

- Training uses the [common GCN flags](presets/training-common.env).
- Apply the [YOLO26n training overrides](presets/training-yolo26n.env) after the common flags. The harness enables BN + SiLU fusion, host prefetch, and the fused target and loss paths.
- Detection applies the [attention and softmax settings](presets/detection-yolo26n.env) and uses the NMS-free CPU selection path with a channels-first head.

The [launcher](../tools/run-radeon-yolo.sh) applies these settings. For example, a single S7150 worker at 512×512 uses the following command.

```bash
tools/run-radeon-yolo.sh s7150 yolo26n detect 512 16 results/yolo26n-512.json
```

## Optimizations and fixes

- GPU candidate selection, a tuned 2,040-wide attention softmax, and a 64×64 attention value GEMM tile with K=32.
- Vectorized attention softmax backward, tiled TAL top-k, fused assignment metrics and target gathers for both heads, and shared target preparation.
- Batch-aware GEMM filter tuning, compact partitioned Winograd weight gradients, and a 2×2 dense stride-2 input-gradient gather.
- Training BN + SiLU fusion across 105 eligible blocks. Backward reconstructs the activation from two per-channel vectors to reduce saved VRAM.
- Fused first-order CIoU + normalized L1 regression for both heads, plus host prefetch. The fusion avoids foreground compaction and reduces dispatches.
- Fixed the invalid tensor layout at batch norm backward by materializing the strided upstream gradient. Two older full-model CPU gradient fixtures still exceed the elementwise tolerance. Final B6 kernel replays passed against the previous GPU path.
- Fused attention, direct masks with the fused top-k mask, early target upload, and the last backward GEMM tiles are outside the final benchmark preset.

Shared platform fixes include Tonga grid splitting for large dispatches, the CPDMA fallback for large clears and copies, and portable reductions. See the [Radeon baseline](RADEON_MINIMAL_BASELINE.md) for runtime background.
