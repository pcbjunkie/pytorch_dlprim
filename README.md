# Pytorch OpenCL backend based on dlprimitives

Radeon development fork: see the [verified five-card baseline](docs/RADEON_MINIMAL_BASELINE.md),
[local changes](LOCAL_CHANGES.md), and the
[experimental legacy Radeon build/profile guide](docs/r600-clover.md).
The upstream network/device results below are retained as upstream documentation;
they are not a validation matrix for this fork's new r600 paths.

## Giving old Radeon GPUs a second life

This project makes older AMD Radeon cards usable for modern PyTorch workloads through OpenCL. It builds on **DLPrimitives and pytorch_dlprim**, with fixes and optimizations across the Torch backend and Mesa’s Clover and Rusticl paths.

The goal is practical: let people experiment with machine learning using hardware they already own. A retired gaming card can become a training device, a wildlife detector, or a way to learn how neural networks work.

### What works today

We’ve trained and run a YOLOv8 nano wildlife detector on Radeon cards ranging from the **HD5870 to the HD7970 and R9 270**. Work included fixing incorrect GPU results, buffer lifetime problems, allocation failures and driver/compiler issues, then improving convolution, reductions, indexing, optimizer updates and synchronization.

At **256×256 FP32**, our completed runs achieved:

| GPU | VRAM | OpenCL path | Training batch | Training images/s | Detection FPS |
|---|---:|---|---:|---:|---:|
| HD5870 | 1 GB | Clover | 24 | 13.02 | 17.40 |
| HD6850 | 1 GB | Clover | 24 | 8.83 | 14.73 |
| HD6970 | 2 GB | Clover | 28 | 14.30 | 15.28 |
| R9 270 | 2 GB | Rusticl | 48 | 21.68 | 32.40 |
| HD7970 | 3 GB | Rusticl | 80 | 29.69 | 27.87 |

These measurements use a **Phenom II X4 with 16 GB RAM**. Detection uses batch one and includes preprocessing and CPU NMS. Every listed configuration completed a sustained training run: 20 epochs for the HD5870, HD6850, HD6970 and R9 270, and 10 epochs for the HD7970 Rusticl result. Training throughput excludes validation and checkpoint writing. These are results for the tested configurations, not matched measurements of hardware efficiency.

The [consolidated Clover baseline](docs/RADEON_MINIMAL_BASELINE.md) records the earlier five-card verification; the GCN Rusticl figures above are subsequent measurements using the private Mesa 25 radeonsi/ACO runtime and the same frozen backend. They require that runtime and the GCN allocator settings (no allocation cache, tight allocation, deferred release, queue limit 128 and retirement budget 64 MiB); the older Rusticl runtime is not equivalent. See the [GCN launcher](tools/with-radeon-gcn-baseline.sh) for the backend settings.

**Memory remains a constraint.** These cards have only 1–3 GB of VRAM, and the legacy OpenCL path has allocation limitations that can prevent training from using all available memory. The legacy Clover configurations use tested memory reservations to improve reliability. Listed batch sizes are verified working settings, rather than universal maximums; larger images and models require smaller batches.

### Who it’s for

- **Students and curious builders** learning PyTorch without buying a new GPU.
- **People reusing old PCs**, particularly where modern hardware is expensive or difficult to obtain.
- **Hobbyists building vision projects** such as wildlife cameras, feeder monitors and small automation systems.
- **Open source developers** interested in improving support for overlooked hardware.

Support is currently demonstrated for specific detection workloads and configurations. This is an experimental backend, and broader model compatibility remains work in progress.

### How it started

The original motivation was a small computer vision project and became an opportunity to experiment with older hardware.

Along the way, it became a demonstration that decade-old GPUs still have useful work left in them.

**Old hardware, modern tools, and a reason to keep learning.**

## Upstream project documentation

DLPrimitives-OpenCL out of tree backend for pytorch

It is only beginning, but you can train some vision nets using OpenCL devices.

Supported pytorch versions are 1.13 and torch `>=` 2.4

# Validated Networks

Following torchvision networks were validated:

| Network               |   Notes                                   |
|-----------------------|-------------------------------------------|
| `alexnet`             |                                           |
| `resnet18`            |                                           |
| `resnet50`            |                                           |
| `convnext_small`      |                                           |
| `vgg16`               |                                           |
| `squeezenet1_0`       |                                           |
| `googlenet`           |                                           |
| `densenet161`         |                                           |
| `inception_v3`        | fwd only - backward fails on cuda/cpu     |
| `shufflenet_v2_x1_0`  |                                           |
| `mobilenet_v2`        |                                           |
| `mobilenet_v3_large`  |                                           |
| `mobilenet_v3_small`  | fwd only - same failure on bwd on cuda.   |
| `resnext50_32x4d`     |                                           |
| `wide_resnet50_2`     |                                           |
| `mnasnet1_0`          |                                           |
| `efficientnet_b0`     |                                           |
| `efficientnet_b4`     |                                           |
| `regnet_y_400mf`      |                                           |
| `vit_b_16`            |                                           |


Calculations validated agaist CPU reference for both forward and backward popogation.

# Tested Devices

DLPrimitves itself is tested on following devies: 

- AMD rx 6600XT with ROCM drivers, rx560 16cu with AMDGPU-pro drivers
- Nvidia: GTX 960
- Intel:  HD 530, UHD 770, Arc A380


# Installation

The simplest way is to download `whl` package from release page - builds for Windows and Linux are provided.

Create virtual evironment, install _CPU_ version of pytorch >= 2.4, download `whl` file that corresponds
to python version, pytorch version and architecture and install it. For example: python 3.10, torch 2.4 on Linux it is:

    pip install pytorch_ocl-0.1.0+torch2.4-cp310-none-linux_x86_64.whl

Now all you need is to import `pytorch_ocl` and now you can use device 'ocl' instead of cuda:

    torch.randn(10,10,device='ocl:0')

If you don't have prebuilt whl file for your enviromnent, you want use latest updates or modify the code,
please refer to README-build.md
    
## How to Use
    
Keep it mind... it is earky version that misses a lot of functionality and it isn't fully tested yet.
So if something fails. It is either not implemented or it is implemented incorrectly

Note: pytorch backend is based on dlprimitives library that actually implements all the operators and
it is relatively well tested.

If you still want to try: import package `pytorch_ocl`

Keep in mind you may have several OpenCL devices. Refer to `clinfo --list` to list
of the devices and their order. Now instead of calling `something.to('cuda')` you call `something.to('ocl:0')` or 
`something.to('privateuseone:0' for pytorch 1.13)` or another `ocl:1` etc.

## Known Issues

1. Many operators not implemented and there may be fallbacks to CPU. Sometimes it is minor but sometimes it may hamper the performance, some may just fail
2. When you save/restore the model move it to CPU. Currently there is an issue with loading back saved state dictionary if it was saved from ocl device


## `pytorch_ocl` specific API

Some functions specific to `pytorch_ocl`. When using pytorch >= 2.4 they are accessible from `torch.ocl` and `pytorch_ocl`, for 1.13 you must use `pytorch_ocl`

- `torch.ocl.empty_cache()`: Same as `torch.cuda.empty_cache()` remove all cached GPU memory
- `torch.ocl.synchronize(device=None)`: synchronize all operations queue on the device, if device is None - all of them same as `torch.cuda.synchonize`
- `torch.ocl.manual_seed_all(seed)`: reset random number generator state. `torch.manual_seed` - it calls automatically for pytorch >= 2.4. Note for pytorch 1.13 you must call `pytorch_ocl.manual_seed_all`


