# YOLOX-nano on Radeon OpenCL

Start with the [GCN build and fixture setup](radeon-yolo.md).

| GPU | Input | Training images/s (batch) | Detection FPS (batch) |
| --- | --- | ---: | ---: |
| R9 270 | 256×256 | 15.38 (B48) | 285.57 (B16) |
| R9 270 | 512×512 | 7.87 (B8) | 101.93 (B8) |
| R9 270 | 1024×1024 | 2.74 (B2) | 31.78 (B4) |
| R9 270 | 1920×1080 | 1.56 (B1) | 18.78 (B4) |
| HD 7970 | 256×256 | 9.94 (B32) | 466.90 (B128) |
| HD 7970 | 512×512 | 5.98 (B16) | 146.62 (B32) |
| HD 7970 | 1024×1024 | 2.43 (B4) | 43.39 (B8) |
| HD 7970 | 1920×1080 | 1.61 (B2) | 26.59 (B8) |
| S7150 ×2 | 512×512 | 33.24 (B16) | 340.81 (B12) |
| S7150 ×2 | 1024×1024 | 14.75 (B8) | 102.05 (B8) |
| S7150 ×2 | 1920×1080 | 10.11 (B7) | 72.71 (B4 × 3) |

S7150 ×2 rates aggregate two physical GPUs. Batches are per worker. B4 × 3 means three batch-4 workers per GPU.
These are fixture throughput measurements. See the [README sweep links](../README.md#what-works-today) for other tested batches.

## Opt-ins

- Training uses the [common GCN flags](presets/training-common.env).
- Apply the [YOLOX-nano training overrides](presets/training-yolox-nano.env) after the common flags. The harness enables BN + SiLU fusion, the optimized assignment geometry, and host prefetch.
- Detection packs BGR input, fuses Focus, and compacts candidates on the GPU before CPU NMS.

The [launcher](../tools/run-radeon-yolo.sh) applies these settings. For example, a single S7150 worker at 512×512 uses the following command.

```bash
tools/run-radeon-yolo.sh s7150 yolox-nano detect 512 12 results/yolox-nano-512.json
```

## Optimizations and fixes

- Fused image packing and Focus, inference Conv/BN folding and Conv + SiLU, tuned pointwise GEMM and width-first depthwise kernels. At HD, three independent batch-4 workers per card kept the detector pipeline busy.
- The existing compact Winograd weight-gradient kernel with 64 partitions replaced the costly stem dispatch.
- Six depthwise 3×3 stride-2 input gradients use gather instead of the generic GEMM path.
- Device comparisons implement the SimOTA center geometry check, avoiding the unsupported dimension-min CPU fallback.
- Training BN + SiLU fusion across 104 eligible blocks reduced saved activations and let batch 7 fit. Host prefetch brought the paired HD rate past 10 images/s.
- The combined B5 replays passed all 330 parameter gradients against the previous GPU path. The B7 timing run passed CPU scalar loss on both cards.

Shared platform fixes include Tonga grid splitting for large dispatches, the CPDMA fallback for large clears and copies, and portable reductions. See the [Radeon baseline](RADEON_MINIMAL_BASELINE.md) for runtime background.
