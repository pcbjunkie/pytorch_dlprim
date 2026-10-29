# Experimental legacy Radeon Clover profile

This is a development snapshot for Mesa's r600 OpenCL path, with correctness
workarounds and opt-in performance changes. Keep the upstream project names and
history. The application, datasets and model checkpoints are separate projects.

## Hardware evidence and pool reservation

| GPU | VRAM | Tested driver/backend | Initial reserve | Evidence |
| --- | ---: | --- | ---: | --- |
| HD 6970 / Cayman XT | 2 GiB | Isolated patched Mesa 25.0.7 Clover, baseline r600 profile (filter target 128, queue 16) | 960 MiB | Completed 30 epochs, batch 24, 256px FP32; finite losses, saved model CPU validation agrees with GPU metrics |
| HD 6850 / Barts | 1 GiB | Earlier patched Clover compatibility experiments | No established recommendation | Small operator checks passed; crashes occurred before later Cayman fixes. Revalidation with the current profile is pending |
| Other HD 68xx | Usually 1 GiB, verify actual card | Not validated with current profile | Unmeasured | Planned |
| HD 58xx / Evergreen | Verify actual card | Not tested with current profile | Unmeasured | Planned |
| R9 270 / Pitcairn | 2 GiB | Mesa 25.0.7 Clover, earlier GCN backend | Not applicable to this r600 global-pool setting | Completed earlier training; do not select the VLIW profile for GCN |

Do **not** carry the 960 MiB reservation to a 1 GiB card. Neither the detected
VRAM size nor one successful allocation establishes a reliable reserve. Desktop
usage, active buffers, driver copies, fragmentation, batch size and kernel
temporaries all matter. The launcher deliberately provides no reservation
default. Set `R600_POOL_RESERVE_MB` only after testing the actual configuration.

For each new card: record PCI/device ID, VRAM, OS/kernel/Mesa/LLVM, profile,
reservation and batch size; start with operator/CPU checks and small batches;
then compare candidate reserves in fresh processes. Record failures and idle,
peak and post-exit memory. A short allocation pass is provisional; recommend a
setting only after sustained real training, finite losses and CPU comparisons.
The 1 GiB tests should establish capacity first, then measure speed separately.

The patched r600 `R600_POOL_RESERVE_MB` requests a contiguous global compute pool
up front, avoiding repeated old/new pool growth copies. It is **not** a Torch
allocation cap or the measured pool size. Valid requests are 1..1024 MiB. Initial
failure falls back to smaller reservations and logs the retry. Later pool growth
can still fail; some old growth-failure paths remain unsafe. A successful
reservation alone does not guarantee a workload fits.

This machine's Linux 6.1 Radeon single-buffer limit is also constrained by its
approximately 1 GiB GTT aperture, despite the 6970 having 2 GiB VRAM. Exactly
1024 MiB was rejected; 960 MiB was usable. A larger GART is a possible separate
kernel/boot experiment, not part of these results. Rusticl shares the relevant
r600 pool machinery and does not automatically remove this limit.

## Build from source

Use the recorded upstream submodule commit. The combined patch includes portable
reductions; **do not also apply** `dlprimitives-portable-reductions.patch`.

```sh
git submodule update --init --recursive
git -C dlprimitives apply --check ../patches/dlprimitives-r600-clover.patch
git -C dlprimitives apply ../patches/dlprimitives-r600-clover.patch
```

The legacy r600/Clover profile requires the patched
[Mesa compatibility fork](https://github.com/pcbjunkie/mesa/tree/r600-clover-25.0),
pinned to commit `0da01a371eb1819211a61bd5e6b506de4d3bca6c`. This is a requirement
for this profile, not for other OpenCL drivers or Squirltool's CPU runtime.
Clover was removed from later Mesa; do not substitute the fork's default `main`
branch or a moving upstream revision.

From the backend repository root:

```sh
git clone --single-branch --branch r600-clover-25.0 https://github.com/pcbjunkie/mesa.git mesa-r600
git -C mesa-r600 checkout --detach 0da01a371eb1819211a61bd5e6b506de4d3bca6c
```

The fork already contains the driver changes: **do not apply the Mesa patch
again**. `patches/r600-bases.json` records the repository, exact commit, upstream
base and original archive SHA256. For an archive-based build instead, the retained
`patches/mesa-25.0.7-r600-clover.patch` applies to pristine Mesa 25.0.7 with
`patch -p1`; this is an alternative to using the fork.

Build an **isolated** Mesa prefix, retaining a known-working system display
driver. The tested Mesa configuration used LLVM 15, r600 and Clover ICD, no
OpenGL/Vulkan/display platforms; a Rusticl build was present but was not used
for the successful training. Relevant Meson options:

```sh
meson setup mesa-r600/build-r600 mesa-r600 --prefix="$PWD/runtime-r600" --libdir=lib --buildtype=release \
  -Dgallium-drivers=r600 -Dgallium-opencl=icd -Dvulkan-drivers= -Dplatforms= \
  -Dopengl=false -Dgles1=disabled -Dgles2=disabled -Degl=disabled \
  -Dglx=disabled -Dgbm=disabled -Dgallium-va=disabled -Dgallium-vdpau=disabled \
  -Dllvm=enabled -Dshared-llvm=enabled -Dbuild-tests=false
ninja -C mesa-r600/build-r600
meson install -C mesa-r600/build-r600
```

Mesa development dependencies and supported build options must match that
version. This is a source-build recipe, not a portable binary package. Standard
Torch wheels may require CPU instructions absent on Phenom II; this machine uses
a separately built Torch 2.4.1 CPU wheel without AVX. The OpenCL backend cannot
repair a host CPU instruction mismatch.

Build the backend following `README-build.md`, with Python/CPU Torch, pybind11,
OpenCL and SQLite development files available. For a typical activated venv:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');$(python -m pybind11 --cmakedir)" \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build -j2
tools/build-r600-shim.sh
```

The shim caps only **unspecified** workgroup sizes at 32. Explicit kernel sizes
and required compiled sizes are retained. It is part of the tested profile,
not a universal statement about hardware workgroup limits.

## Run without workspace-specific paths

Create an ICD directory containing only a `clover.icd` file pointing to the
isolated prefix's `lib/libMesaOpenCL.so.1`. Export `OCL_ICD_VENDORS` to that
directory and `R600_MESA_PREFIX` to the prefix. The launcher derives repository
paths and accepts normal commands; `DLPRIM_PYTHON_PATH` may select an installed
backend instead of the repository's `build` directory.

```sh
OCL_ICD_VENDORS=/path/to/clover-icd R600_MESA_PREFIX=/path/to/runtime-r600 \
  R600_POOL_RESERVE_MB=960 tools/with-r600-clover.sh \
  python tests/local/check_memory_reporting.py
```

That reserve is the **6970 setting above**, not a suggestion for 1 GiB hardware.
The prefix must contain the patched r600 pipe as well as Clover. If keeping
separate prefixes, explicitly set `GALLIUM_PIPE_SEARCH_DIR` to the patched pipe
directory. Never assume a system-installed Clover picked up a local patch.

## Selected behavior and tuning

`tools/with-r600-clover.sh` consolidates the previously chained development
wrappers. It enables gather convolution/pooling gradients, BN workgroups capped
at 64, small GEMM tiles, packed byte outputs, branchless legacy indexing and flat
copies, sorted indexed reduction, fused SiLU backward, four-pixel input gradients,
atomic-free split filter gradients, tight no-cache allocation and bounded
deferred buffer retirement. Driver color-cache flush is required before relevant
pool copies. Original upstream defaults remain available without the launcher.

Deferred buffers remain alive behind actual GPU completion markers. The selected
queue limit is 32 helper calls and retirement threshold 32 MiB; these are bounded
tuning controls, not hardware thread counts. Override `OPENCL_ASYNC_OP_LIMIT`
or `OPENCL_DEFERRED_RELEASE_MB` for measured experiments. Larger values can
increase memory pressure. `DLPRIM_FILTER_SPLIT_GROUPS` is an experimental
power-of-two group target, 64..1024, retaining the 64-partition/4-MiB workspace
bounds; invalid settings retain the default of 128. It is not a local size.
The launcher selects target 256 after bounded CPU checks and matched benchmarks.
The completed 30-epoch baseline used queue 16 / target 128; the final tuning has
not yet repeated that long run. See [performance evidence](r600-performance.md).

The standalone reproducers in `tests/local/r600` deliberately include known
failing/hanging controls. Read their README; do not blindly run every historical
test under one profile. The final BN workgroup is 64, not the older experimental
single-wave/128 tests. Full ATen native-BN saved-statistic compatibility is still
an audit item. General odd-spatial convolution failures remain tracked; fast
paths have explicit eligibility guards. FP16, all layouts/dtypes, multi-GPU
training and untested GPUs are not established by this work.

`tests/local/run_r600_checks.py` runs a selected current-profile suite in separate
processes, covering copies, output offsets, unique/topk, indexed reduction,
SiLU, filter/input gradients, deferred lifetimes and memory reporting. Use
`--list` to inspect it. It excludes historical hang controls and is not a
complete dtype/layout/API audit or a replacement for a sustained training run.

## Memory reporting

Torch allocator APIs retain their meanings: live requested tensor storage,
rounded backend reserved storage (including cached and deferred buffers), and
peaks since startup/explicit reset. Views do not count their backing storage twice.
These counters exclude Mesa pool capacity, driver temporaries and other processes.

`examples/ultralytics_memory.py` provides an optional trainer mixin for
Ultralytics 8.3.0. Put `OpenCLMemoryMixin` before the normal trainer in the MRO.
It changes the progress field to `OCLpeak_GB` (decimal GB, high-water backend
reserved storage), and writes detailed byte counters to `opencl-memory.jsonl`
once per epoch. It does not reset peaks or synchronize/poll the GPU per step.
The requested reserve is recorded separately; unmeasured driver-pool and
whole-card counters are null. Use independent driver telemetry such as radeontop
for whole-card residency. A small between-batch allocator value is not proof of
available pool capacity.
