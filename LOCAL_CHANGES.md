# Radeon OpenCL development snapshot

This fork retains Artyom Beilis's project name and upstream history. It is an
experimental source snapshot, not a general hardware-support claim or an
upstream-reviewed release. The wildlife app, data and model weights are separate.

The earlier R9 270 / Mesa 25.0.7 Clover profile completed 100 epochs. The baseline
HD 6970 / patched r600 Clover profile has now completed 30 epochs from random
weights at batch 24, 256px FP32. Losses remained finite; CPU validation of the
saved best model agreed with GPU validation at displayed precision. A separate
post-run gradient check passed (cosine 0.999982, norm ratio 0.999658). This is
evidence of stable training, not a claim of adequate model accuracy or bitwise
CPU/GPU equality. HD 68xx and HD 58xx current-profile testing remains pending.
The final filter-partition/queue tuning has bounded CPU/benchmark checks;
the completed 30-epoch run used target 128 and queue limit 16.

## Main changes

See [changes by layer](docs/changes-by-layer.md) for module ownership and the
distinction between reproduced bugs, algorithm changes and legacy workarounds.

- Device discovery/selection and per-device live/reserved/cache/peak accounting.
- Scalar promotion, repeat/view fallback and pointwise destination-offset fixes.
- GPU norms, batched clipping/SGD/EMA, masking, scatter, advanced indexing,
  addmv, bounded small-k topk, small-list unique and max pooling with indices.
- Legacy 32-bit indexing/division and packed-byte-store compatibility paths;
  conservative explicit workgroups and standalone correctness reproducers.
- Atomic-free gather input/pooling gradients and split filter gradients;
  vectorized input gradients, fused SiLU backward and sorted indexed reduction.
- Tight no-cache allocation plus bounded deferred retirement behind GPU markers.
- Optional meaningful YOLO allocator-peak reporting, separate from driver VRAM.

See [the r600 build/profile guide](docs/r600-clover.md) for relocatable launch
commands, settings, per-card pool-reservation evidence and memory meanings.
[Benchmark evidence](docs/r600-performance.md) records the measured improvements.
[Historical investigation notes](docs/r600-history.md) retain intermediate
failures and provisional conclusions; their old 'training disabled/pending'
statements describe those earlier snapshots, not the final selected profile.

## Reproducible source dependencies

The upstream dlprimitives gitlink is unchanged. All modified runtime sources,
including new kernels, are shipped in `patches/dlprimitives-r600-clover.patch`.
Apply it to the recorded submodule base; it already includes the older portable
reductions patch. The legacy profile pins the separate
[Mesa compatibility fork](https://github.com/pcbjunkie/mesa/tree/r600-clover-25.0)
at `0da01a371eb1819211a61bd5e6b506de4d3bca6c`. The retained Mesa patch is an
alternative for pristine 25.0.7 archives, not an extra patch to apply to the fork.
Exact bases and the conditional dependency are in `patches/r600-bases.json`.
No unpublished submodule commit, local binary, dataset, credential or model
checkpoint is needed to recover the code.

## Limits before upstream submission

Some fast paths support only the dtypes/layouts/shapes used by these models;
other cases keep existing fallbacks and diagnostics. Unique rank sorting is
bounded to 4096 values and legacy topk selection favors k <=32. Native BN's
saved-statistic public ATen contract, general view/out/storage bounds and
separate odd-spatial convolution failures still need broader audit. Mesa's
later pool-growth OOM handling and GTT/single-buffer limits are unresolved.

The workgroup and subword-store workarounds have reproducible evidence, but
their exact compiler/driver/hardware root causes are not all established.
Multi-GPU enumeration exists; OpenCL DataParallel/DDP training is not validated.
No FP16 or untested-card support is promised. Keep correctness fixes, driver
patches and performance work in separate focused upstream proposals later.
