"""Comparable per-epoch training timings for OpenCL, CUDA, or CPU trainers."""
import json
from pathlib import Path
import time


def attach_training_telemetry(trainer):
    import torch
    starts = {}
    def synchronize(t):
        if t.device.type == 'cuda':
            torch.cuda.synchronize(t.device)
        elif t.device.type == 'ocl':
            torch.ocl.synchronize(t.device)
    def begin(t):
        synchronize(t)
        starts['epoch'] = time.perf_counter()
    def training_end(t):
        synchronize(t)
        seconds = time.perf_counter()-starts['epoch']
        images = len(t.train_loader.dataset)
        row = {'epoch':t.epoch+1, 'training_seconds':seconds,
               'training_images':images, 'training_images_per_second':images/seconds,
               'batch_size':t.args.batch, 'image_size':t.args.imgsz,
               'workers':t.args.workers, 'device':str(t.device),
               'precision':'FP32', 'includes':'data loading, augmentation, forward/loss/backward, optimizer/EMA and batch logging; excludes validation/checkpoint writing'}
        with (Path(t.save_dir)/'training-timing.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
    trainer.add_callback('on_train_epoch_start',begin)
    trainer.add_callback('on_train_epoch_end',training_end)
