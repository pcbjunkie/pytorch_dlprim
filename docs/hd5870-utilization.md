# HD5870 training and inference utilization, 2026-10-29

A subsequent [larger training-batch study](hd5870-large-batch.md) extends this
initial batch 1/2/4/8 sweep. Its capacity results and changed reservations are
recorded separately.

Cypress XT 1 GiB, headless Phenom II X4 955/16 GiB, patched Mesa 25.0.7 Clover, frozen pool32-v1 backend. YOLO nano five-class checkpoint, 256px FP32, automatic workgroup cap 32, BN cap 64, queue 64. The required per-kernel `clFlush` workaround remains enabled. No Torch/Mesa library changes or arithmetic optimizations in this study.

## Native training

Each candidate passed two separate CPU loss/full-gradient comparisons, then a fresh process measured 16 optimizer updates after 6 warmups. One loader worker, fused EMA, 512 MiB initial pool reservation. No CPU comparisons, profiling queue or API tracing inside these timed windows. These are short real-data probes, not full epochs.

| Batch | Images/s | Mean batch ms | Backend peak reserved MiB | Radeontop mean / peak |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.720 | 1389.7 | 64.1 | Not captured |
| 2 | 1.080 | 1851.2 | 98.6 | 15.0% / 41.7% |
| 4 | 2.299 | 1740.2 | 167.0 | 23.1% / 42.5% |
| 8 | 4.998 | 1600.5 | 301.5 | 37.9% / 61.7% |

Radeontop samples are one second apart and selected within native timing windows; batch 1 predates the monitor. Backend bytes describe live/deferred buffer accounting, not the full initial driver reservation or whole-card VRAM. Batches above 8 were not tested; no memory-capacity conclusion follows from this sweep.

## Native inference

Same fixed 20-image manifest/checkpoint as the matched batch-one protocol; separate batch harness. Each mode uses 20 warmups and 100 measured batches. CPU raw prediction and NMS checks run before and after timing with unchanged tolerances. Last groups wrap earlier images to retain a full batch. All candidates passed. Pool reserve 128 MiB.

| Batch | Resident forward images/s | Decoded frames → detections images/s | Mean batch ms | P95 batch ms | CPU image comparisons |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16.09 | 15.51 | 64.49 | 84.62 | 800 |
| 2 | 29.55 | 26.92 | 74.29 | 94.21 | 800 |
| 4 | 43.61 | 36.61 | 109.26 | 131.03 | 800 |
| 8 | 57.99 | 43.77 | 182.78 | 197.12 | 960 |

Full frame pipeline includes CPU preprocessing, upload/normalization, network, readback, NMS, box scaling and detection dictionaries. It excludes camera capture, JPEG decode, HTTP, drawing and disk reads. Batch latency is time to process the whole batch, not a lower single-camera response time. The earlier generic matched batch-one result (17.43 FPS decoded) is retained; this sweep measured a separate repeat/harness.

## Separate diagnostic profiles

Training has 6 natural steps, 5 phase-barrier steps and 3 event/API-instrumented steps. Inference has 5 instrumented full-pipeline batches. GPU timestamps and host timestamps have separate origins. Coverage below is the union of captured device-event intervals divided by instrumented wall time; gaps may include untraced work. This is not ALU occupancy. API unions avoid double-counting nested host calls, but API time still overlaps GPU execution and must not be added to device time.

| Workload | Batch | Instrumented batches | Wall ms | All captured event coverage | Named non-transfer coverage | Host enqueue ms | Host flush ms | Explicit blocking API ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 1 | 3 | 3953.3 | 16.0% | 15.1% | 1354.0 | 841.8 | 177.6 |
| inference | 1 | 5 | 505.2 | 41.1% | 35.1% | 124.3 | 132.6 | 17.5 |
| training | 8 | 3 | 4567.8 | 39.9% | 38.8% | 1589.9 | 945.1 | 389.3 |
| inference | 8 | 5 | 1022.4 | 77.0% | 59.5% | 178.3 | 170.1 | 240.1 |

The non-transfer column excludes events named `read`, `write`, `copy` and `marker`. Transfer event durations on this driver are not a measurement of shader activity. Neither coverage column establishes hardware occupancy or an exact idle fraction.

### Training phase barriers, batch 8

Diagnostic phase means in milliseconds per update; barriers alter overlap and these are not native throughput estimates.

| Phase | Mean ms |
| --- | ---: |
| data_loading | 3.32 |
| preprocess_and_input_transfer | 3.89 |
| network_forward | 196.28 |
| detection_loss | 355.30 |
| backward | 642.11 |
| gradient_unscale | 0.51 |
| gradient_clipping | 34.17 |
| sgd_update | 60.52 |
| scaler_update | 0.26 |
| zero_grad | 6.67 |
| ema_update | 23.17 |
| loss_logging_host_read | 0.44 |
| total | 1326.85 |

### Training, batch 1: largest captured device events

| Kernel or transfer | Calls in window | Event ms per batch |
| --- | ---: | ---: |
| batched_pointwise | 285 | 50.26 |
| topk_select | 3 | 40.03 |
| conv_gemm | 189 | 32.08 |
| conv_filter_partials | 75 | 16.66 |
| conv_input_gather4_3x3 | 78 | 16.17 |
| read | 120 | 12.48 |
| conv_gemm_bwd_filter | 120 | 10.85 |

### Inference, batch 1: largest captured device events

| Kernel or transfer | Calls in window | Event ms per batch |
| --- | ---: | ---: |
| conv_gemm | 320 | 32.67 |
| read | 5 | 6.06 |
| pointwise_exec | 325 | 1.04 |
| slice_copy | 255 | 0.56 |
| softmax | 5 | 0.44 |
| max_pool_with_indices32 | 15 | 0.24 |
| pointwise_exec_broadcast | 90 | 0.17 |

Host phase means in milliseconds; inclusive driver calls can include waits:

| Phase | Mean ms |
| --- | ---: |
| preprocess_upload_host | 3.31 |
| forward_dispatch_host | 91.14 |
| output_readback_host | 3.14 |
| nms_scaling_dictionaries_host | 3.02 |
| wall_seconds | 100.60 |

### Training, batch 8: largest captured device events

| Kernel or transfer | Calls in window | Event ms per batch |
| --- | ---: | ---: |
| conv_filter_partials | 147 | 117.34 |
| conv_input_gather4_3x3 | 78 | 101.36 |
| conv_gemm | 189 | 100.20 |
| batched_pointwise | 285 | 50.20 |
| conv_input_gather | 126 | 43.96 |
| topk_select | 3 | 40.43 |
| conv_gemm_bwd_filter | 54 | 21.28 |

### Inference, batch 8: largest captured device events

| Kernel or transfer | Calls in window | Event ms per batch |
| --- | ---: | ---: |
| conv_gemm | 320 | 102.68 |
| read | 5 | 35.70 |
| pointwise_exec | 325 | 6.95 |
| softmax | 5 | 3.47 |
| slice_copy | 255 | 3.36 |
| max_pool_with_indices32 | 15 | 1.58 |
| copy_strided | 255 | 1.51 |

Host phase means in milliseconds; inclusive driver calls can include waits:

| Phase | Mean ms |
| --- | ---: |
| preprocess_upload_host | 15.44 |
| forward_dispatch_host | 152.82 |
| output_readback_host | 22.19 |
| nms_scaling_dictionaries_host | 13.58 |
| wall_seconds | 204.03 |

## Interpretation and next targets

Batching raises throughput substantially and increases measured utilization. At small batches this workload spends substantial time submitting many short operations. Training loss/indexing and optimizer bookkeeping also contribute fixed per-update work; data loading is only a few milliseconds in these profile windows. Backward convolution remains the largest captured GPU component at batch 8.

The main next investigation is safe submission/launch amortization: isolate why Cypress requires per-kernel flushing and reduce redundant work only with strict CPU parity tests. Removing the flush flag is not an optimization supported by these results: its absence already produced wrong predictions. Host API time may include driver stalls, so these intervals do not establish a speedup ceiling.

A narrower training target is the top-k selector: about 40 ms of captured device time per update at both batch 1 and8. Backward filter/input kernels are the main arithmetic targets at batch 8. Inference profiles distinguish convolution work from CPU preprocessing/postprocessing; multi-image throughput should not be confused with live-camera latency. Larger workgroups and batches remain separate experiments.

The root follow-new dmesg capture remained active and empty. These short tests do not replace a long training stability run. A first training profile harness failed because of a missing multiprocessing main guard; it was fixed and the diagnostic profiles rerun in new directories. The failure is retained and does not invalidate the independently completed native sweeps.

[Machine-readable results](benchmarks/hd5870-utilization-20261029.json), [initial card results](hd5870-results.md). Local raw logs/events/traces and controllers: `runs/hd5870-utilization-20261029/`; training profile data: `runs/hd5870-train-phase-b{1,8}-r2-20261029/`.
