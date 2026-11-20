# Wildlife YOLOv8n 512×512 training on one S7150

FP32, GPU 0 (07:00.0), one loader worker. Each run starts from the same wildlife checkpoint and uses the real augmented dataset loader, four warmup updates and sixteen measured updates. Timing includes loading, input upload, forward, loss, backward, gradient clipping, SGD and EMA. No validation epoch was timed.

| Batch | Training images/s | Peak sampled VRAM | Peak sampled GPU temperature |
|---:|---:|---:|---:|
| 8 | 18.94 | 1.87 GiB | 56 °C |
| 16 | 22.51 | 3.26 GiB | 56 °C |
| 20 | 24.96 | 3.79 GiB | 58 °C |
| 24 | 24.92 | 4.19 GiB | 59 °C |
| 32 | 26.34 | 5.15 GiB | 60 °C |
| 40 | 27.38 | 6.32 GiB | 60 °C |

Every batch passed the first-batch CPU loss comparison and finite gradient checks throughout. The highest measured rate was batch 40. Batch 20 used about 40% less sampled VRAM at 91% of its throughput. Batch 20 at 512×512 matches the total image pixels per update of batch 80 at 256×256. This does not establish the same memory requirement or rate on a different device.
