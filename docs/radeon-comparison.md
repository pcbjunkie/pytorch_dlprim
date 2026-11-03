> Current training, inference and stability results are consolidated in
> [Radeon benchmarks](RADEON_BENCHMARKS.md). This page retains the earlier
> theoretical-FP32 comparison and historical measurements.

# Reference FP32 peak and observed inference

Peak arithmetic is calculated from reference shader count × maximum reference
clock GHz × 2 operations per multiply-add. It is not measured sustained throughput;
factory-overclocked cards and actual runtime clocks may differ.

| Card | Shader count | Reference MHz | Peak FP32 GFLOPS | Decoded-frame inference FPS |
| --- | ---: | ---: | ---: | ---: |
| [HD6850](https://www.pugetsystems.com/parts/Video-Card/Asus-Radeon-HD-6850-1GB-7269/) | 960 | 775 | 1,488 | 15.52 |
| [HD5870](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/programmer-references/AMD_OpenCL_Programming_Optimization_Guide2.pdf) | 1600 | 850 | 2,720 | 17.43 |
| [HD6970](https://www.pugetsystems.com/parts/Video-Card/Sapphire-Radeon-HD-6970-2GB-7386/) | 1536 | 880 | 2,703 | Not yet measured with this suite |
| [R9 270](https://ir.amd.com/news-events/press-releases/detail/463/amd-radeon-r9-270-graphics-card-packs-a-1080p-gaming-punch) | 1280 | 925 | 2,368 | Not yet measured with this suite |

FPS uses the same YOLO nano five-class checkpoint and20image manifest,256px
FP32,batch1,and includes preprocessing/transfers/CPU NMS. Training throughput
is a separate workload. HD5870 uses an additional per-kernel submission flush;
profiles/boots/CPU baseline timing differ. Therefore this is not a clean
hardware-only scaling experiment.

HD5870 has about83% more theoretical FP32 arithmetic thanHD6850, but its measured
frame-processing throughput is about12% higher. This indicates peak arithmetic
alone is a poor predictor for this workload. The figures do not isolate which
CPU/driver/kernel/memory/compiler cost dominates.

The AMD R9 270 release lists1280shaders and925MHz but prints inconsistent2.69TFLOPS;
the table uses1280×0.925×2=2368GFLOPS rather than that headline value.

[Machine-readable record](benchmarks/reference-fp32-vs-inference-20261029.json).

The [HD5870 utilization study](hd5870-utilization.md) now records separate
training and inference batch sweeps and GPU-event/host-API profiles. Native
training reaches 5.00 images/s at batch 8 in a 16-update probe. Decoded-frame
inference reaches 43.77 images/s at batch 8, compared with 15.51 at batch 1 in
that sweep. Batch 8 latency is 182.78 ms per batch; this is aggregate throughput,
not 43.77 FPS for a single low-latency webcam stream. These observations show
batch amortization matters, while the profiles also identify submission overhead
and convolution costs. The matched batch-one table above remains unchanged.

The [larger HD5870 training-batch follow-up](hd5870-large-batch.md) reaches
9.81 images/s at batch 20 / reserve 768 MiB and 10.37 at batch 24 / reserve
960 MiB, with separate CPU loss/gradient checks. The latter is about 5.8% faster;
the curve continues improving beyond batch 20 with diminishing returns. These
short runs do not establish a pure arithmetic bottleneck or a controlled
hardware comparison with the HD6850's different checkpoint/profile trials.
