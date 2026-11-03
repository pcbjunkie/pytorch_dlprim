# Current consolidated results

The completed 2026-11-03 baseline and per-card runtime settings are documented in
[RADEON_MINIMAL_BASELINE.md](RADEON_MINIMAL_BASELINE.md) and the
[machine-readable release record](benchmarks/radeon-baseline-release-v1.json).
All five models passed 20 added epochs. The phase-one table below is preserved
as historical evidence and is not the latest result table.

# Radeon training and inference benchmarks

Living comparison, updated 2026-11-01. This is the overall results document;
linked per-card reports retain configurations, failures, profiles and raw data.
Current baseline order: HD5870, HD6850, HD6970, R9 270, HD7970. Further performance
optimization and possible Rusticl comparisons follow this baseline pass.

## Earlier phase-one baseline pass

YOLOv8 nano, five wildlife classes, 256px FP32, stock Ultralytics 8.3.0.
Host: Phenom II X4 955, 16 GiB RAM, Debian 12, headless. Each card uses its
verified driver/backend configuration; those differences must accompany results.

| Card | VRAM | Training images/s | Training batch | Forward-only FPS | Decoded-frame FPS | Encoded-image FPS | Stability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HD5870 | 1 GiB | 12.27 overall; 12.54 final five epochs | 24 | 16.95 | 16.56 | 15.11 | 20 added epochs, 3h10m, saved-checkpoint CPU validation passed |
| HD6850 | 1 GiB | 8.35 | 24 | 17.36 | 15.55 | 14.68 | 20 epochs completed, 4h29m; 14 TTM placement warnings, no captured reset/stall |
| HD6970 | 2 GiB | 14.22 | 28 | 16.75 | 15.25 | 14.21 | 20 epochs completed, 2h44m; one TTM placement warning, no captured reset/stall |
| R9 270 | 2 GiB | 15.29 | 48 | 11.73 | 12.56 | 13.34 | 20 epochs completed, 2h23m; no new kernel messages |
| HD7970 | 3 GiB | 20.59 | 80 | 9.60 | 11.54 | 13.91 | 20 epochs completed, 1h47m; no captured GPU errors; power setup qualified |

HD5870 training: frozen parallel-topk V8 backend, one loader worker, reservation
960 MiB, queue128, retirement64 MiB, required per-kernel flush workaround.
Training-only time includes loading, augmentation, forward/loss/backward,
optimizer/EMA and normal batch logging; validation and checkpoint time are excluded.
The final five epochs are a subset of the same run, not an independent trial.
40 TTM placement warnings occurred and execution continued. No GPU reset,
lockup, ring stall or failed CS relocation was captured. This is a completed
run with recorded allocation warnings, not a warning-free driver audit.

HD5870 inference: same V8 backend, reservation128 MiB, queue64, batch1, fused
convolution/BN, two Torch threads and one OpenCV thread. Uses the frozen original
checkpoint and SHA-checked 20-image suite, rather than the newly trained model.
Each of three modes has20 warmups and100 timed frames. All800 CPU/raw-output/NMS
comparisons passed; no new kernel messages. Decoded-frame mean60.37ms,
p95 79.11ms; all300 timed samples across the three modes were under100ms.
The final training shutdown's CPU validation timing is not a GPU benchmark.

## Earlier results retained for context

These use earlier snapshots/configurations and are not matched against the
current HD5870 baseline. Pending cells above are deliberately not filled with
older numbers.

| Card | Earlier evidence | Detailed record |
| --- | --- | --- |
| HD5870 | Earlier matched decoded-frame17.43FPS; separate batch sweep15.51FPS at batch1,43.77 images/s at batch8 | [Bring-up](hd5870-results.md), [utilization](hd5870-utilization.md), [top-k](hd5870-topk.md) |
| HD6850 | Completed7 plus20 training epochs; matched decoded-frame15.52FPS | [Card results](hd6850-results.md) |
| HD6970 | Earlier30-epoch training averaged12.297 images/s; later bounded tuned probe12.756 images/s | [Performance](r600-performance.md) |
| R9 270 | Completed earlier100-epoch wildlife training; historical speed estimates need reconciliation with saved timing windows | [History](r600-history.md) |

Batch8 inference images/s measures aggregate multi-image throughput, not
single-camera frame latency. Earlier theoretical-FP32 comparisons and their
source links remain in [reference arithmetic and historical inference](radeon-comparison.md).

## Measurement and evidence

Keep model/data hashes and measurement settings fixed for matched comparisons.
Report common-batch training separately from the practical throughput at each
card's selected batch. Use [the inference protocol](inference-benchmarks.md),
including CPU parity checks outside timing. Record software/binary hashes,
workgroups, queues, memory reservations and kernel warnings with each result.
Neither shader GFLOPS nor radeontop busy percentages substitute for throughput.

[Machine-readable overall record](benchmarks/radeon-baselines-20261030.json).
Local raw HD5870 artifacts: `runs/hd5870-wildlife-resume20-20261030/` and
`runs/hd5870-final-inference-20261030/` in the experiment workspace.

## Batch and memory allocation records

Record each attempted batch, requested pool reservation, allocator live/reserved
peaks, retirement budget, throughput, allocation failures, and time-aligned
whole-card VRAM/GTT samples. Requested reservation is not measured pool capacity;
backend counters exclude the driver pool and other processes. Whole-card VRAM
is measured separately with radeontop when available. Failed attempts remain
part of the capacity record; largest tested batch is not an exact maximum.

HD5870 completed training: batch24, requested reservation960MiB, peak backend-reserved 888036424 bytes (846.90MiB), retirement budget64MiB. HD6850 current retest is recording batches12/20/24 with requested reservations512/768/960MiB in separate processes. Results remain pending until those stages complete.

HD6850 V8: training reservation960MiB, batch24, peak backend-reserved 898812304 bytes. Full trial/allocation and VRAM/GTT evidence: `runs/hd6850-baseline-v8-20261030/`; stability: `hd6850-wildlife-v8-stability20-20261030`.

HD6850 final review: training averaged8.355 images/s, settled near8.443 images/s. Stability peak backend-reserved898812304bytes (857.17MiB), requested reservation960MiB. Saved checkpoint CPU validation passed. Fourteen TTM placement warnings occurred during stability with continued execution; no captured resets, lockups or ring stalls. Batch28 capacity attempt separately failed buffer allocation and terminated with SIGBUS; this is not graceful OOM recovery. Ready for HD6970 baseline.

HD6970 V8: training reservation1008MiB, batch28, peak backend-reserved 1028562064 bytes. Full trial/allocation and VRAM/GTT evidence: `runs/hd6970-baseline-v8-20261030/`; stability: `hd6970-wildlife-v8-stability20-20261030`.

HD6970 final review: batch28, requested preallocation1008MiB, training14.222 images/s over20epochs, elapsed2h44m including validation/checkpoints/finalCPUevaluation. Peak backend-reserved1028562064bytes (980.91MiB). Inference decoded15.25FPS,800CPUcomparisons passed. Savedbest reloaded and CPUvalidated. One TTM placement warning during stability; execution continued with no captured reset/lockup/ringstall. Batch29 passed but slower; batch30 failed allocation and terminated withSIGBUS. Baseline complete, ready forR9270. Multi-buffer experiment remains deferred; no new GPU tests started.

R9 270 current allocator check: 30 independent64MiB Torch buffers,1920MiB
live simultaneously; GPU read/modify/write via add_ and full host verification
passed for every element while all buffers remained live. Whole-card samples
peaked at1938.14MiB VRAM and69.59MiB GTT. Allocated/reserved backend counters
returned to zero after release/empty_cache; no new kernel messages. Driver
reports2GiB total and512MiB maximum single allocation. This used the existing
GCN allocator without r600 pool reservations or arena changes; it does not
resolve the separate r600 memory limitation. Results: `runs/r9-270-allocation-20261030/`.

R9 retest qualification: initialV8 top-k failed onradeonsi. A V9 local-storage
layout fix passes operator regressions, but default-cache batch24 still fails
full-model parity. The resumed baseline uses V9 with no-cache/tight allocation
and deferred event retirement (queue128/64MiB), which passed two full-model
CPU updates each atbatch2 and24. This differs from the initial1920MiB allocation
probe's V8/default allocation profile; keep those results scoped separately.
Current run: `runs/r9-270-baseline-v9-retired-20261030/`.

R9 270 V9: training uses normal GCN allocation (r600 reservation not applicable), batch48, peak backend-reserved 1733772912 bytes. Full trial/allocation and VRAM/GTT evidence: `runs/r9-270-baseline-v9-20261030/`; stability: `r9-270-wildlife-v9-retired-stability20-20261030`.

R9 270 final review: completed20epochs atbatch48 in2h23m08s including validation
and checkpoint/finalCPUevaluation. Training-only average15.2888images/s, final
five15.3862images/s. Peak backend-reserved1733772912bytes; dynamicGCNallocation,
no r600 preallocation. Savedbest reloaded and CPUvalidated. Kernel capture
added0bytes during stability. Finalepoch mAP50/mAP50-95=0.69602/0.42041.

R9 uses correctedV9 and explicit deferred retirement; earlier legacy-card
baselines usedV8, so these are verified per-card configurations rather than a
pure hardware-only comparison. Inference passed800checks: forward11.73FPS,
decoded12.56FPS (79.63ms mean,117.62ms p95), encoded13.34FPS. Independent timing
windows produced an unusual mode ordering; preserve results and repeat matched
profiling/old-backend comparisons in phase two rather than assuming decoding
is beneficial. Batch56-96 completed but slowed; backend allocations above2GiB
do not prove VRAM residency. [Detailed R9 report](r9-270-results.md).

HD7970 identification: Tahiti XT,3GiB. Allocation gate passed with2880MiB
live across45buffers, GPU read/modify/write and fullCPUreadback verification.
Sampled peaks: VRAM2900.05MiB/GTT69.64MiB; backendcounter returned
tozero afterrelease. SameV9/deferred profile asR9baseline, no r600preallocation.
Benchmark/20epoch pass now inprogress; resultsremainpending. Allocation evidence
`runs/hd7970-allocation-20261030/`; pass `runs/hd7970-baseline-v9-retired-20261030/`.

HD7970 pass interrupted: user confirmed a 6-pin PCIe plug in an 8-pin GPU socket, with no 8-pin/6+2 lead available on this PSU. GPU load stopped before 20-epoch stability. Initial allocation, CPU checks and partial batch measurements are preserved with this power-configuration qualification; they are not a completed baseline with the intended power connections. User subsequently said it will most likely be fine, without explicitly requesting resumption.

HD7970 V9: training uses normal GCN allocation (r600 reservation not applicable), batch80, peak backend-reserved 2818098820 bytes. Full trial/allocation and VRAM/GTT evidence: `runs/hd7970-baseline-v9-20261030/`; stability: `hd7970-wildlife-v9-retired-stability20-20261030`.

## Completed first baseline round

All five cards completed their recorded20-epoch passes: HD5870, HD6850,
HD6970, R9 270 and HD7970. Software/configuration differences and recovered
allocation warnings remain documented; this is not a hardware-only ranking.
The HD7970 completed atbatch80 in1h46m56s, averaging20.590images/s, finalfive
20.558images/s. Savedbest reloaded and CPUvalidated. Kernel capture contains
only an unrelated cupsd AppArmor audit event, with no captured GPU reset,
stall or allocation warning during stability. Power connection remains
qualified: user reports Sapphire Dual-X/two8-pin sockets with a6-pin PSU plug
in an8-pin socket; exact remaining PSU-plug layout is unconfirmed. No power
consumption or absence-of-throttling claim is made.

HD7970 peak backend-reserved2818098820bytes, without r600 preallocation.
Inference800CPUcomparisons passed; decoded11.54FPS remains a phase-two target.
[Detailed HD7970 report](hd7970-results.md). Next: restoreHD5870 as primary
development card, first run pendingV9top-k regressions, then phase-two
optimization. CUDA comparison stays at the end of phase two.

## HD5870 V9 regression — 2026-11-01

The shared top-k local-storage fix passed 183 focused/general cases, two
batch-24 full-model CPU loss/gradient comparisons, and 800 fixed-suite
inference comparisons on Cypress. No new kernel messages were captured.
The frozen V8 baseline above remains unchanged.

Matched short native probes (batch24, reservation960 MiB, one worker,
queue128, retirement64 MiB, six warmups and24 measured steps): V8 control A
10.712 images/s, V9 11.964, V8 control B 12.025. The control spread is large
enough that this does not establish a V9 throughput improvement.
V9 inference: forward18.653 FPS, decoded-frame17.646 FPS (mean56.670 ms,
p95 61.234 ms), encoded-image15.514 FPS.

Raw artifacts: `runs/hd5870-v9-regression-20261101/`. This is a short
correctness/performance regression, not a repeated20-epoch V9 stability
qualification. HD6850/HD6970 V9 regression checks remain pending.
