# Optional Ultralytics OpenCL example

This adapter is separate from the backend library. It uses Ultralytics **8.3.0**
and the caller's existing model and YOLO-format dataset YAML. Install the matching
Torch/Torchvision CPU environment first; then install the optional Ultralytics
dependency into that environment. These examples do not download datasets or
bundle weights. Ultralytics and model/dataset licensing are separate from this
backend's license; check their terms for your use.

The adapter selects FP32, keeps network execution on OpenCL, performs NMS/metrics
on CPU, saves CPU-portable FP32 checkpoints, and uses the backend's fused EMA
when eligible. It reports `OCLpeak_GB`, detailed allocator counters per epoch,
and training-only per-epoch timings. No operator profiler or per-batch CPU
gradient comparisons run during ordinary training.

With the recorded patched driver and tested 6970 profile configured:

```sh
R600_POOL_RESERVE_MB=960 tools/with-r600-clover.sh \
  python examples/train_yolo_opencl.py --data /path/to/data.yaml \
  --model /path/to/yolov8n.pt --task detect --epochs 30 --imgsz 256 \
  --batch 24 --workers 1 --name my-run
```

Pool and batch values above are the 2 GiB 6970 example, **not** settings validated
on a 1 GiB card. Start a new GPU with small operator checks and its own capacity
tests. `--device` defaults to `ocl:0`; multi-GPU training is not implemented.
`--task pose` preserves the existing pose adapter but has not received the same
full legacy-card training validation as the detection path.

`--model` accepts an existing checkpoint or architecture YAML. The example's
conservative SGD learning rate defaults to 0.001 and can be changed with `--lr0`;
the appropriate learning-rate schedule and training duration depend on whether
you are fine-tuning or starting from random weights. The prior 30-epoch scratch
run was undertrained; it is a stability result, not a recommended quality recipe.

For a bounded throughput benchmark with the same fixed checkpoint/data:

```sh
R600_POOL_RESERVE_MB=960 tools/with-r600-clover.sh \
  python examples/train_yolo_opencl.py --data /path/to/data.yaml \
  --model /path/to/fixed-checkpoint.pt --imgsz 256 --batch 24 --workers 1 \
  --epochs 1 --warmup-steps 4 --probe-steps 24 --name bounded-benchmark
```

The dataset must provide enough full batches in one epoch for the requested
warm-up and measurement window. The benchmark saves `benchmark.json`, skips
checkpoint/initial-model saving and stops before validation. It does not alter
the supplied checkpoint. Compared configurations must use the same binary,
checkpoint, data/augmentation settings, batch and measurement window; repeat
the control to assess variation. No speed claim follows from GPU utilization
alone. Outputs default to `runs/opencl`; `--project` changes that directory.

`--resume /path/to/last.pt` retains ordinary resume behavior and is deliberately
incompatible with the bounded benchmark option. This example is pinned to the
tested Ultralytics internals; newer releases may require adapter changes.
