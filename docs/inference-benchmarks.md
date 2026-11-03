# Per-card YOLO inference protocol

Run this protocol on every card. Keep the checkpoint, image manifest, software
snapshot and measurement parameters fixed for comparisons. Use each card's
CPU-checked compatibility profile and record any configuration differences.

[The optional example](../examples/benchmark_yolo_inference.py) accepts local
YOLO detection weights and a SHA256-checked image manifest. It supports CPU,
OpenCL and CUDA without a web server or changes to Squirltool. The tested
environment uses Torch 2.4.1 and Ultralytics 8.3.0.

## Measurements

Batch 1, 256×256 FP32, evaluation mode; fuse convolution/BN on CPU before moving
the network to its target device. Use two Torch threads and one OpenCV thread.
MKLDNN/NNPACK are disabled for the Phenom-compatible CPU comparison; this is
not a benchmark of an optimized inference engine on a modern CPU.

Each mode has 20 warm-up frames and 100 timed frames. Report mean, median and
95th-percentile latency, reciprocal-mean FPS and the fraction within 100 ms.
GPU work is synchronized so dispatch time cannot substitute for completion.

| Mode | Included |
| --- | --- |
| Device-resident forward | Network and YOLO head decode on a prepared input; no host transfers or NMS |
| Decoded frame to detections | BGR resize/letterbox, channel conversion, tensor upload/normalization, network, output readback, CPU NMS, box scaling and detection dictionaries |
| Encoded image to detections | Decode preloaded image bytes, then the complete decoded-frame pipeline |

Load files before timing. Camera capture, video decode, disk access, HTTP,
drawing and output-image encoding are excluded. This measures processing
latency, not a complete webcam or Flask load test. GPU allocator counters are
process counters; a driver-pool reservation is not live tensor usage.

## Correctness gate

Compare raw predictions and final NMS boxes against CPU. The default GPU run
performs 20 passes over the image manifest before timing and another 20 after:
800 comparisons for a 20-image suite. CPU references/assertions are untimed.

Raw box tolerances: `rtol=0.002, atol=0.002`; scores:
`rtol=0.003, atol=0.000001`. Final detection counts/classes must agree, matched
IoU must be at least 0.99 and confidence differences no more than 0.002.
Default confidence/IoU thresholds are 0.25/0.7, maximum 100 detections.
Failed raw comparisons save input/CPU/GPU tensors and the failing image/check.
Do not relax tolerances to obtain a speed result.

HD6850 tests showed that automatic workgroup caps 64/128/256 could pass single
images but fail repeated fused batch-1 raw predictions. Keep the generic r600
wrapper's conservative automatic cap 32 for inference. Explicit kernel sizes
are separate from that cap. The exact failing kernel/compiler cause is not
isolated. This is separate from the completed training/CPU checkpoint validation.

Cap 32 is a starting profile, not proof of correctness on another GPU. Cypress
HD5870 testing exposed a queued fused-inference difference even at cap 32.
The tested workaround calls `clFlush` after each successful kernel enqueue,
without inserting `clFinish`. Build `tools/build-r600-flush-shim.sh` and enable
`R600_FLUSH_AFTER_KERNEL=1` before the legacy launcher when testing this profile.
Record that flag as a configuration difference; keep the failing unmodified
trial and repeat the full CPU gate. The exact underlying driver cause remains
unresolved. See [the HD5870 results](hd5870-results.md) for the bounded evidence.

## Freeze a suite

Choose representative images and preserve them alongside the checkpoint.
The manifest uses paths relative to `--images`:

```json
{
  "schema_version": 1,
  "model_sha256": "SHA256 of checkpoint",
  "images": [{"path": "frame01.jpg", "sha256": "SHA256 of frame01.jpg"}]
}
```

A simple local manifest can be generated as follows; review class coverage:

```python
import hashlib, json
from pathlib import Path

model = Path('/path/to/best.pt')
images = Path('/path/to/images')
selected = sorted(p for p in images.iterdir()
                  if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})[:20]
assert selected
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'schema_version': 1, 'model_sha256': digest(model),
            'images': [{'path': p.name, 'sha256': digest(p)} for p in selected]}
Path('inference-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
```

The local wildlife suite has four distinct validation images per class, 20
total, and the HD6850 20-epoch checkpoint. Change the suite version when
changing images or weights; do not overwrite comparison inputs.

## Run

```sh
python examples/benchmark_yolo_inference.py --device cpu \
  --model /path/to/best.pt --images /path/to/images \
  --manifest inference-manifest.json --output runs/inference/cpu.json
```

With the isolated patched Clover/Mesa environment configured according to
[the legacy guide](r600-clover.md), an HD6850-style bounded trial is:

```sh
R600_POOL_RESERVE_MB=128 OPENCL_ASYNC_OP_LIMIT=64 \
  tools/with-r600-clover.sh python examples/benchmark_yolo_inference.py \
  --device ocl:0 --model /path/to/best.pt --images /path/to/images \
  --manifest inference-manifest.json --output runs/inference/ocl.json
```

The reservation is workload-specific, not a cross-card default. Keep desktop
state fixed, run timed trials sequentially, retain a live kernel-log capture
and preserve failures/interrupted trials. Record model/manifest/script/backend
hashes, device/runtime identity, compatibility flags and memory counters.
Report camera-processing FPS separately from training images/s.
