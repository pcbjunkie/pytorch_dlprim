# Radeon YOLO setup and batch presets

The GCN benchmarks use FP32 on a private Mesa RustiCL runtime with radeonsi.
Choose a model below for tested batch sizes and its optimization notes.

| Model | GCN cards tested | Configuration |
| --- | --- | --- |
| YOLOv8n wildlife | R9 270, HD 7970, FirePro S7150 | [Batches and opt-ins](yolov8n-s7150-config.md) |
| YOLO26n | R9 270, HD 7970, FirePro S7150 | [Batches and opt-ins](yolo26n-s7150-config.md) |
| YOLOX-nano | R9 270, HD 7970, FirePro S7150 | [Batches and opt-ins](yolox-nano-s7150-config.md) |

For HD 5870, HD 6850 and HD 6970, use the [verified Clover release and card presets](RADEON_MINIMAL_BASELINE.md). They cover YOLOv8n at 256×256. The GCN opt-ins need fresh validation on those GPUs.

## Build and select the runtime

From the repository root with the submodule initialized, apply the source patch to the pinned DLPrimitives revision before configuring CMake. Use a fresh submodule checkout for this build.

```sh
git submodule update --init --recursive
git -C dlprimitives apply --check ../patches/dlprimitives-radeon-gcn.patch
git -C dlprimitives apply ../patches/dlprimitives-radeon-gcn.patch
```

Then follow the [source build instructions](../README-build.md). The GCN patch
includes the portable reductions. Apply only this DLPrimitives patch for the
GCN build. For S7150, the [R720 Tonga patch](../patches/mesa-r720-tonga.patch)
targets Mesa commit `0da01a371eb1819211a61bd5e6b506de4d3bca6c` in the
[project Mesa fork](https://github.com/pcbjunkie/mesa). Build it into a private
prefix. For R9 270 and HD 7970, the qualified runtime uses Mesa 25.0.7
radeonsi/ACO and the regular Radeon kernel driver. Select the private ICD and
its library directory before running the launcher.

## Run a configured case

The [launcher](../tools/run-radeon-yolo.sh) takes a GPU preset, model, workload,
resolution, batch, and output JSON path. It selects the [shared training flags](presets/training-common.env)
and the model overrides listed in each guide. It launches one worker. Use
`--device ocl:1` for a different enumerated device and `--pci-device` on
detection to select optional driver memory counters.

```bash
export OPENCL_POSE_ROOT=/path/to/benchmark-workspace
export DLPRIM_PYTHON_PATH=/path/to/pytorch_dlprim/build
export OCL_ICD_VENDORS=/path/to/private-icd-directory
export LD_LIBRARY_PATH=/path/to/private-mesa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHON=/path/to/venv/bin/python
tools/run-radeon-yolo.sh hd7970 yolox-nano detect hd 8 results/hd7970-yolox-hd.json
```

Use CPU PyTorch 2.4.1 and torchvision 0.19.1, with NumPy and OpenCV in the
environment. The benchmarked Ultralytics versions were 8.3.0 for YOLOv8n and
8.4.214 for YOLO26n and YOLOX-nano. The YOLOX example also uses `loguru` and
`tabulate`. Provide these inputs beneath `OPENCL_POSE_ROOT`.

| Path | Input |
| --- | --- |
| `fixtures/wildlife-256/manifest.json` | JSON with an `images` array. Each entry has `path` relative to the images directory and `sha256`. |
| `fixtures/wildlife-256/images/` | Decoded frame fixtures in image files. |
| `datasets/squirrel-wildlife-v2/valid/labels/` | YOLO text labels with matching stems, five classes, normalized center coordinates and box dimensions. Training only. |
| `fixtures/wildlife-256/model.pt` | Five-class YOLOv8n wildlife checkpoint. |
| `models/hd-comparison/yolo26n.pt` | Ultralytics YOLO26n checkpoint. |
| `models/hd-comparison/yolox_nano.pth` | Official YOLOX-nano checkpoint. |
| `sources/YOLOX/` | Megvii YOLOX source checkout at `6ddff4824372906469a7fae2dc3206c7aa4bbaee`. |

Bring your own weights and fixtures. The repository includes the benchmark
code and flags. Training on the published benchmark inputs used controlled
fixture updates. Detection starts from decoded frames and includes
postprocessing. HD pads 1920×1080 to 1920×1088. Batch sizes are per worker.
The published S7150 pair rates aggregate concurrent workers on two GPUs.

## Charts

- [Training by GPU, resolution and model](benchmarks/charts/radeon-train.pdf)
- [Detection by GPU, resolution and model](benchmarks/charts/radeon-detect.pdf)
- [HD 7970 training batch sweeps](benchmarks/charts/hd7970-train-batch.pdf)
- [HD 7970 detection batch sweeps](benchmarks/charts/hd7970-detect-batch.pdf)
