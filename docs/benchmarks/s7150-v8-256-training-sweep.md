# Current YOLOv8n 256px training sweep

| GPU0 batch | Training images/s | Peak VRAM GiB |
|---:|---:|---:|
| 32 | 66.595 | 1.738 |
| 64 | 79.763 | 3.176 |
| 80 | 82.354 | 3.617 |
| 96 | 85.626 | 4.078 |
| 128 | 88.883 | 4.994 |
| 160 | 87.497 | 6.118 |
| 192 | 90.490 | 7.092 |

| Batch | Training images/s | CPU check |
|---:|---:|---|
|128|91.978|First-batch loss passed|
|128 repeat|92.349|All201 parameter gradients passed|
|192|No measured result|Raw-gradient gate stopped before optimizer update|
