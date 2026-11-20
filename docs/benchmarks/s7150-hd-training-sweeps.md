# YOLOv8n full-HD training batch resweep

| GPU0 batch | Training images/s | Peak VRAM GiB |
|---:|---:|---:|
| 1 | 2.620 | 1.651 |
| 2 | 3.359 | 3.075 |
| 3 | 3.645 | 3.889 |
| 4 | 3.961 | 4.763 |
| 5 | 4.015 | 5.851 |
| 6 | 4.157 | 6.962 |

In the single-GPU sweep, a guarded batch 7 attempt stopped before timing. At a
higher guard limit, batch 7 measured 4.078 and 4.119 images/s with 16 and 64 MiB
deferred release, respectively.

## Later dual-GPU training sweep

Rates aggregate two independent workers. Batch sizes and peak memory are per GPU.

| Batch per GPU | Training images/s | Peak VRAM per GPU |
| ---: | ---: | ---: |
| 4 | 9.523 | 5.00 GiB |
| 5 | 9.573 | 6.17 GiB |
| 6 | 10.001 | 7.09 GiB |
| 7 (64 MiB deferred release) | 9.312 | 7.81 GiB |

## Final YOLO26n training sweep

Single S7150, five-class head initialized from compatible COCO weights, labeled HD fixtures. Four warmups and 12 measured SGD updates per batch. Timing includes in-memory packing and upload, forward, loss, backward, finite checks, clipping and SGD. Every completed batch passed the initial CPU scalar loss check.

| Batch/GPU | Training images/s | Peak sampled VRAM | Result |
|---:|---:|---:|---|
| 1 | 1.56 | 2.21 GiB | Completed |
| 2 | 2.06 | 3.85 GiB | Completed |
| 3 | 2.33 | 5.11 GiB | Completed |
| 4 | 2.67 | 6.54 GiB | Completed |
| 5 | 2.59 | 7.66 GiB | Completed |
| 6 | — | 7.80 GiB | VRAM guard before timing, initial sweep |

## YOLOX-nano early single-GPU sweep

| Batch per GPU | One-card images/s | Peak GiB | Outcome |
| --- | ---: | ---: | --- |
| 1 | 1.96 | 1.77 | Completed |
| 2 | 2.55 | 3.14 | Completed |
| 4 | 3.20 | 5.77 | Completed |
| 5 | 3.31 | 7.07 | Completed |
| 6 | — | 7.70 | VRAM guard |

Batch 5 reached 6.89 images/s across both cards over 160 images in 23.21 s. Peak sampled VRAM was 7.12 GiB on one card and 7.07 on the other. The separate profile ran at 3.40 images/s in its uninstrumented window.
