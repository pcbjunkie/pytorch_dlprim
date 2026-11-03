# HD5870 backward-pass follow-up, 2026-10-29

Headless Cypress XT / 1 GiB, Phenom II X4 955 / 16 GiB. Same fixed five-class YOLO nano checkpoint, 256px FP32, batch 20, one loader worker, automatic cap 32 / BN cap 64, queue 64 and per-kernel flush workaround. Initial reservation is 768 MiB. No Torch core or Mesa driver changes in this study.

## Native training

Each probe measures 24 updates after six warmups. CPU comparisons and profiling are outside native timing. The two original controls bracket the first tile/filter-split comparisons; the tile-only control was repeated immediately before the final depthwise candidate.

| Configuration | Images/s | Delta vs original control mean | Peak backend reserved bytes |
| --- | ---: | ---: | ---: |
| control-256-a | 10.34405 | +0.43% | 740943400 |
| tile-256 | 10.42679 | +1.24% | 740943400 |
| split-128 | 10.30068 | +0.01% | 740943400 |
| split-512 | 10.25561 | -0.42% | 740943400 |
| control-256-b | 10.25445 | -0.43% | 740943400 |
| tile-256-repeat | 10.39764 | +0.96% | 740943400 |
| depthwise-tuned-256 | 10.75937 | +4.47% | 740943400 |

The original control mean is 10.29925 images/s. The final candidate reaches 10.75937 images/s: +4.47% against that mean, or +3.48% against the immediate tile-only repeat at 10.39764. The original controls differ by about 0.87%; these short runs have no statistical confidence interval. All timed candidates have the same recorded peak backend reservation. Whole-card VRAM is a different measure.

## Separate matched profiles

These profiles use a profiling queue, phase barriers and event/CPU tracing. They locate costs; their phase totals must not replace native throughput. Device-event times overlap host phase timings and must not be added to them. All values below are milliseconds per batch of 20.

| Component | Original | Tile only | Tile + depthwise |
| --- | ---: | ---: | ---: |
| Full backward, phase-barrier wall time | 1127.86 | 1113.27 | 1044.23 |
| Backward convolution, captured device events | 677.37 | 666.56 | 598.13 |
| 3x3 input gradients, captured device events | 248.60 | 236.45 | 236.52 |
| Generic partial filter gradients, captured device events | 310.49 | 311.65 | 232.49 |
| Specialized depthwise filter gradients, captured device events | 0.96 | 0.96 | 11.69 |

The combined change reduces captured backward-convolution device time about 11.7% and the separately measured full backward phase about 7.4%. Filter reduction/other calls are retained in the JSON. Ordinary dense filter gradients and input gradients remain substantial costs.

## What changed

1. Narrow stride-two input gradients use a 4x16 local group instead of 8x8 when the four-pixel path has input width 16 and height at least 16. Both groups contain 64 work items. The affected standalone kernel measured 33.047 to 22.735 ms at batch 24; other shapes retain 8x8. A broad layout replacement and eight-pixel vectorization regressed important shapes and were rejected.
2. The split-filter override previously selected generic matrix multiplication before considering the existing specialized depthwise reduction. The new selection preserves the specialized path for compatible one-output-per-input-channel depthwise convolutions. Channel multipliers, ordinary grouped convolutions and unsupported stride/dilation configurations retain generic handling.
3. The legacy depthwise reduction no longer starts dozens of nearly empty groups per coefficient. Its power-of-two partition count is bounded by work per lane. Existing first-stage 256-thread groups remain; second-stage sizes vary with the partition count. The existing DLPRIM_FILTER_SPLIT_K opt-in controls the legacy scheduling/index handling; no new user flag is introduced.
4. The existing depthwise kernel returned wrong gradients on a 17x19 map on the frozen baseline too. The legacy path now uses exact unsigned coordinate division, with host-specialized image dimensions allowing power-of-two shifts to fold at compilation. Nonlegacy coordinate handling remains unchanged. Baseline failure tensors and the passing fixed reproducer are retained. No specific upstream compiler-source cause is established.

The original two largest inferred generic depthwise calls each cost about 28.14 ms. Five depthwise calls hidden in the generic filter events sum to about 79.25 ms. This shape attribution matches the deterministic CPU convolution sequence to device events across separate windows, checking all 67 partial-filter calls per update; it is an inference without shared correlation IDs. Including the previously specialized calls, the baseline depthwise total is about 80.21 ms, versus about 11.71 ms including the new second-stage reductions.

## Correctness and source distribution

The tile change passed all 24 existing input-gradient cases three times and two full-model CPU loss/gradient comparisons. Both tested generic filter split targets and the final depthwise candidate each passed two separate full-model CPU comparisons at batch 20. Tolerances were unchanged.

New `tests/local/check_depthwise_filter_selection.py` covers 12 cases with three full-tensor CPU comparisons each, nonzero tensor offsets, dispatch assertions, public autograd accumulation, one/two-stage reductions, kernels 1/3/5, odd maps and grouped/channel-multiplier/stride/dilation fallbacks. Run it through the current legacy profile with `--output /path/to/new/test-artifacts`. The original odd-map case fails on the unchanged baseline and passes with the final candidate using the same fixed seed.

Changes belong to DLPrimitives `src/core/conv.cpp` and `src/kernels/depthwise_separable_bw_filter.cl`, plus the backend regression test. The combined [DLPrimitives patch](../patches/dlprimitives-r600-clover.patch) reconstructs all 17 runtime files exactly from the recorded upstream base; its hash and frozen binary/source hashes are retained in the JSON. The original backend/inference snapshots remain available. Local `with-hd5870-backward.sh` selects the new frozen training profile. Other-card retesting and long training remain separate work.

## Kernel audit and limits

Recovered TTM placement/eviction warnings occurred during the input-gradient regression and full-model CPU-reference checks. The continuous root capture was retained; no GPU reset/lockup/stall was observed. Native benchmark/profile stages added no kernel messages in this window. The study is not warning-free or a long-duration stability proof. No long training, checkpoint replacement, inference retest, commit or push was performed.

An initial small C benchmark harness had a shadowed status variable; those first results were discarded. Corrected r2 sources compiled with warnings treated as errors and passed all 14 shape cases with full GPU-output comparison and independent CPU samples. Later depthwise attempts, baseline odd-map failure and corrected passing runs are retained separately.

[Machine-readable results](benchmarks/hd5870-backward-20261029.json), [card results](hd5870-results.md), [earlier utilization study](hd5870-utilization.md). Local logs, sources and controllers: `experiments/hd5870-backward-20261029/`.

## How workgroup shapes are chosen

Torch dispatches the tensor operation to the OpenCL backend. DLPrimitives selects an algorithm and kernel parameters, compiles or retrieves the OpenCL program, and supplies a local size at launch. This gather kernel uses global worker coordinates, so 8x8 and 4x16 use the same OpenCL kernel; the host chooses between them. The driver may perform internal compilation/specialization as well. Other kernels bake tile sizes into compilation parameters or require a fixed local size and cannot be reshaped independently. The compatibility shim only chooses sizes for kernels that leave local size unspecified; it does not override an explicit gather tile.
