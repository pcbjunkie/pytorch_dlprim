"""Repeatable batch-one FP32 YOLO detection latency and CPU parity benchmark.

Uses local weights and a SHA256-checked image manifest. No downloads, camera,
web server, drawing, disk I/O, operator profiler or CPU checks in timed loops.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64)
    return dict(count=len(samples), mean_ms=float(values.mean() * 1000),
        median_ms=float(np.median(values) * 1000),
        p95_ms=float(np.percentile(values, 95) * 1000),
        min_ms=float(values.min() * 1000), max_ms=float(values.max() * 1000),
        frames_per_second=float(1 / values.mean()),
        fraction_at_or_below_100ms=float((values <= .1).mean()),
        samples_seconds=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='ocl:0')
    parser.add_argument('--label', default='local')
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--check-repeats', type=int, default=20,
                        help='GPU parity passes over the manifest, outside timing')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.7)
    parser.add_argument('--max-det', type=int, default=100)
    args = parser.parse_args()
    if args.device.startswith('ocl'):
        import pytorch_ocl  # Register the optional backend before torch.device.
    device = torch.device(args.device)
    assert args.imgsz > 0 and args.imgsz % 32 == 0
    assert args.iterations > 0 and args.warmup > 0 and args.check_repeats > 0
    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled = False
    torch.backends.nnpack.set_flags(False)
    cv2.setNumThreads(1)
    torch.manual_seed(2026)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    assert manifest['images']
    if 'model_sha256' in manifest:
        assert sha256(args.model) == manifest['model_sha256'], 'Changed benchmark checkpoint'
    encoded, frames = [], []
    for item in manifest['images']:
        path = args.images / item['path']
        assert sha256(path) == item['sha256'], f'Changed benchmark image: {path}'
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        assert image is not None, path
        encoded.append(data)
        frames.append(image)
    resize = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False)

    def synchronize():
        if device.type == 'ocl':
            torch.ocl.synchronize(device)
        elif device.type == 'cuda':
            torch.cuda.synchronize(device)

    def prepare(frame, target):
        image = resize(image=frame)
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        return torch.from_numpy(image).unsqueeze(0).to(target).float().div_(255)

    def prediction(model, tensor):
        value = model(tensor)
        return value[0] if isinstance(value, (list, tuple)) else value

    setup = time.perf_counter()
    reference = YOLO(str(args.model)).model.cpu().float().eval()
    reference.fuse(verbose=False)
    names = reference.names
    names = dict(enumerate(names)) if isinstance(names, list) else dict(names)
    network = reference if device.type == 'cpu' else copy.deepcopy(reference).to(device)
    synchronize()
    model_setup_seconds = time.perf_counter() - setup

    def postprocess(raw, shape):
        # NMS mutates predictions, so preserve reference data used by parity.
        boxes = non_max_suppression(raw.clone(), conf_thres=args.confidence,
            iou_thres=args.iou, nc=len(names), max_det=args.max_det)[0]
        if len(boxes):
            boxes[:, :4] = scale_boxes((args.imgsz, args.imgsz), boxes[:, :4], shape[:2])
        height, width = shape[:2]
        return [dict(class_id=int(label), label=names[int(label)], confidence=float(score),
            xyxy=[float(x1), float(y1), float(x2), float(y2)],
            xyxy_normalized=[float(x1 / width), float(y1 / height),
                             float(x2 / width), float(y2 / height)])
            for x1, y1, x2, y2, score, label in boxes.tolist()]

    def detection_parity(expected, actual):
        assert len(expected) == len(actual), (expected, actual)
        available = list(actual)
        minimum_iou, max_confidence_error = 1., 0.
        for wanted in expected:
            candidates = []
            a = np.asarray(wanted['xyxy'])
            for index, found in enumerate(available):
                if found['class_id'] != wanted['class_id']:
                    continue
                b = np.asarray(found['xyxy'])
                intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
                union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
                candidates.append((float(intersection / max(union, 1e-12)), index))
            assert candidates, (wanted, available)
            overlap, index = max(candidates)
            found = available.pop(index)
            confidence_error = abs(wanted['confidence'] - found['confidence'])
            assert overlap >= .99 and confidence_error <= .002, (wanted, found, overlap)
            minimum_iou = min(minimum_iou, overlap)
            max_confidence_error = max(max_confidence_error, confidence_error)
        return minimum_iou, max_confidence_error

    result = dict(schema_version=1, label=args.label, device=str(device),
        model_sha256=sha256(args.model), manifest_sha256=sha256(args.manifest),
        script_sha256=sha256(__file__), image_count=len(frames), image_size=args.imgsz,
        input_shapes=[list(frame.shape) for frame in frames], batch=1, precision='FP32',
        fused_convolution_batchnorm=True, cpu_threads=args.threads, opencv_threads=1,
        warmup_per_mode=args.warmup, iterations_per_mode=args.iterations,
        requested_gpu_check_repeats=args.check_repeats,
        confidence=args.confidence, iou=args.iou, max_detections=args.max_det,
        torch_version=torch.__version__, python_version=platform.python_version(),
        model_setup_seconds=model_setup_seconds, names=names,
        environment={key: os.environ.get(key) for key in [
            'R600_POOL_RESERVE_MB', 'OPENCL_ASYNC_OP_LIMIT', 'DLPRIM_BN_MAX_WG',
            'DLPRIM_INDEX_WG', 'DLPRIM_FILTER_SPLIT_GROUPS', 'DLPRIM_FLAT_STRIDED_COPY']},
        measurement_scope='Sequential batch-one latency. GPU work is synchronized; '
            'CPU parity checks are outside timing. Encoded files are preloaded; '
            'no camera/video capture, disk reads, HTTP, drawing or JPEG output encoding.',
        correctness=[], timings={})
    if device.type == 'ocl':
        result['device_name'] = torch.ocl.get_device_name(device)
    elif device.type == 'cuda':
        result['device_name'] = torch.cuda.get_device_name(device)
    else:
        result['device_name'] = platform.processor()

    with torch.inference_mode():
        cpu_inputs = [prepare(frame, 'cpu') for frame in frames]
        # Cache CPU references once; stress GPU allocation/dispatch reuse before
        # timing instead of accepting one successful pass over the images.
        references = [prediction(reference, tensor).detach() for tensor in cpu_inputs]
        repeats = args.check_repeats if device.type != 'cpu' else 1
        def verify(phase):
            for check in range(repeats * len(frames)):
                index = check % len(frames)
                frame, tensor, expected = frames[index], cpu_inputs[index], references[index]
                synchronize()
                start = time.perf_counter()
                actual = prediction(network, tensor.to(device)).detach().cpu()
                synchronize()
                if check == 0 and phase == 'before_timing':
                    result['first_prediction_upload_forward_readback_seconds'] = time.perf_counter() - start
                try:
                    assert bool(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(actual[:, 4:], expected[:, 4:], rtol=3e-3, atol=1e-6)
                except (AssertionError, RuntimeError):
                    torch.save(dict(image=manifest['images'][index],
                        input=tensor, expected=expected, actual=actual),
                        args.output.with_suffix('.failure.pt'))
                    args.output.with_suffix('.failure.json').write_text(json.dumps(dict(
                        image=manifest['images'][index], index=index, check=check, phase=phase,
                        max_box_absolute_error=float((actual[:, :4] - expected[:, :4]).abs().max()),
                        max_score_absolute_error=float((actual[:, 4:] - expected[:, 4:]).abs().max()),
                        preceding_checks=result['correctness']), indent=2) + '\n')
                    raise
                expected_boxes = postprocess(expected, frame.shape)
                actual_boxes = postprocess(actual, frame.shape)
                overlap, confidence_error = detection_parity(expected_boxes, actual_boxes)
                row = dict(image=manifest['images'][index]['path'], check=check, phase=phase, detections=len(actual_boxes),
                    max_box_absolute_error=float((actual[:, :4] - expected[:, :4]).abs().max()),
                    max_score_absolute_error=float((actual[:, 4:] - expected[:, 4:]).abs().max()),
                    minimum_matched_detection_iou=overlap,
                    max_detection_confidence_error=confidence_error)
                result['correctness'].append(row)
                if (check + 1) % len(frames) == 0:
                    print('PARITY', check + 1, '/', repeats * len(frames), flush=True)
            print('PASS', len(result['correctness']), 'CPU/raw-output/NMS checks on',
                  len(frames), 'real images', flush=True)
        verify('before_timing')
        device_inputs = [tensor.to(device) for tensor in cpu_inputs]
        synchronize()

        def frame_pipeline(frame):
            tensor = prepare(frame, device)
            raw = prediction(network, tensor).detach().cpu()
            return postprocess(raw, frame.shape)

        def forward(index):
            return prediction(network, device_inputs[index])

        def decoded_frame(index):
            return frame_pipeline(frames[index])

        def encoded_image(index):
            return frame_pipeline(cv2.imdecode(encoded[index], cv2.IMREAD_COLOR))

        for mode, operation in [('device_resident_forward', forward),
                                ('decoded_frame_to_detections', decoded_frame),
                                ('encoded_image_to_detections', encoded_image)]:
            print('BENCHMARK', mode, flush=True)
            for index in range(args.warmup):
                output = operation(index % len(frames))
                synchronize()
                del output
            samples = []
            for index in range(args.iterations):
                synchronize()
                start = time.perf_counter()
                output = operation(index % len(frames))
                synchronize()
                samples.append(time.perf_counter() - start)
                del output
            result['timings'][mode] = summarize(samples)
            print(mode, json.dumps({key: value for key, value in result['timings'][mode].items()
                                    if key != 'samples_seconds'}), flush=True)
        verify('after_timing')
    if device.type == 'ocl':
        result['allocator'] = torch.ocl.memory_stats(device)
    result['correctness_passed'] = True
    result['cpu_comparison_is_cross_device'] = device.type != 'cpu'
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT', args.output, flush=True)


if __name__ == '__main__':
    main()
