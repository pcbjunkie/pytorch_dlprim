# Optimized full-HD inference batch resweep

| Batch/GPU | V8 combined FPS | V8 VRAM GiB | YOLO26 combined FPS | YOLO26 VRAM GiB |
|---:|---:|---:|---:|---:|
| 1 | 30.335 | 0.458 | 26.994 | 0.522 |
| 2 | 32.361 | 0.728 | 29.778 | 0.786 |
| 4 | 36.387 | 1.335 | 32.916 | 1.373 |
| 8 | 38.953 | 2.403 | 35.046 | 2.320 |
| 12 | 38.183 | 3.524 | 34.469 | 3.390 |
| 16 | 38.947 | 4.135 | 34.601 | 4.430 |
| 24 | 38.961 | 5.222 | 34.819 | 6.125 |
| 32 | 38.737 | 6.368 | 35.302 | 7.446 |
| 40 | 39.107 | 7.078 | not attempted | — |

| Model | Batch/GPU | Combined inference FPS | Peak VRAM/GPU GiB | Measured seconds |
|---|---:|---:|---:|---:|
| wildlife | 8 | 38.7617 | 2.589 | 60.266 |
| wildlife | 40 | 38.9712 | 7.166 | 61.584 |
| yolo26 | 8 | 34.9472 | 2.609 | 60.434 |
| yolo26 | 32 | 35.2896 | 7.446 | 61.661 |

## Later YOLOv8n dual-GPU sweep

| Batch per GPU | Combined FPS | Peak VRAM per GPU |
| ---: | ---: | ---: |
| 1 | 32.36 | 0.66 GiB |
| 8 | 40.85 | 2.90 GiB |
| 16 | 40.75 | 4.57 GiB |
| 40 | 40.76 | 7.48 GiB |

## Final YOLO26n inference sweep

COCO checkpoint. Two S7150s with one worker per GPU, NMS-free GPU selection and final detection readback. All cases passed 20-fixture CPU checks on each worker before timing and a repeated CPU check after timing.

| Batch/GPU | Combined FPS | GPU 0 / GPU 1 FPS | Peak VRAM/GPU | Measured frames / time |
|---:|---:|---:|---:|---:|
| 1 | 29.97 | 14.98 / 15.15 | 0.71 GiB | 601 / 20.06 s |
| 4 | 36.49 | 18.28 / 18.33 | 1.80 GiB | 736 / 20.17 s |
| 8 | 38.57 | 19.38 / 19.31 | 2.69 GiB | 1,744 / 45.21 s |
| 16 | 38.79 | 19.63 / 19.40 | 4.76 GiB | 800 / 20.62 s |
| 24 | 38.51 | 19.42 / 19.26 | 6.17 GiB | 816 / 21.19 s |

Batch 16 had the highest short-run throughput. Batch 8 is within 0.6% and has a longer timed window and about 2.07 GiB less peak VRAM per GPU.
