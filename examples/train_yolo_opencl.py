"""Ultralytics 8.3.0 trainer adapter for the local DLPrimitives backend."""
import argparse
from copy import copy, deepcopy
from datetime import datetime
import io
import os
from pathlib import Path

import torch
import pytorch_ocl
import ultralytics
from fused_ema import OpenCLModelEMA
from ultralytics_memory import OpenCLMemoryMixin
from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator


def atomic_checkpoint(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def to_cpu(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def save_initial_checkpoint(trainer):
    # Preserve the exact initialization for later CPU/backend comparisons.
    model = deepcopy(trainer.model).cpu().float()
    if hasattr(model, 'criterion'):
        del model.criterion
    if isinstance(getattr(model, 'args', None), dict):
        model.args = to_cpu(model.args)
    elif hasattr(model, 'args') and hasattr(model.args, 'device'):
        model.args.device = str(model.args.device)
    checkpoint = {'epoch':-1, 'model':model, 'ema':None, 'updates':0,
                  'train_args':to_cpu(vars(trainer.args)), 'version':ultralytics.__version__}
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    atomic_checkpoint(Path(trainer.save_dir)/'initial.pt', buffer.getvalue())


class OpenCLValidationMixin:
    """Keep neural-network execution on GPU and run NMS/metrics on CPU."""
    def postprocess(self, preds):
        prediction = preds[0] if isinstance(preds, (tuple, list)) else preds
        return super().postprocess(prediction.detach().cpu())

    def update_metrics(self, preds, batch):
        saved_device = self.device
        self.device = torch.device('cpu')
        try:
            return super().update_metrics(preds, to_cpu(batch))
        finally:
            self.device = saved_device


class OpenCLPoseValidator(OpenCLValidationMixin, PoseValidator):
    pass


class OpenCLDetectionValidator(OpenCLValidationMixin, DetectionValidator):
    pass


class OpenCLTrainerMixin(OpenCLMemoryMixin):
    fused_ema = True

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        if self.fused_ema and self.ema is not None:
            # Keep the already created/restored EMA weights, update count,
            # decay schedule, and metadata; only replace the update method.
            self.ema.__class__ = OpenCLModelEMA

    def get_validator(self):
        if self.args.task == 'pose':
            self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
            validator = OpenCLPoseValidator
        else:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
            validator = OpenCLDetectionValidator
        return validator(self.test_loader, save_dir=self.save_dir,
                         args=copy(self.args), _callbacks=self.callbacks)

    def _clear_memory(self):
        torch.ocl.empty_cache()

    def save_model(self):
        # Preserve FP32 weights and CPU optimizer state. Upstream saves FP16
        # tensors on the original device, which this old GPU cannot execute.
        ema = deepcopy(self.ema.ema).cpu().float()
        # Criterion caches are rebuilt when training resumes. They can retain
        # device objects/tensors even after Module.cpu() has moved parameters.
        if hasattr(ema, 'criterion'):
            del ema.criterion
        if isinstance(getattr(ema, 'args', None), dict):
            ema.args = to_cpu(ema.args)
        elif hasattr(ema, 'args') and hasattr(ema.args, 'device'):
            ema.args.device = str(ema.args.device)
        checkpoint = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': None,
            'ema': ema,
            'updates': self.ema.updates,
            'optimizer': to_cpu(self.optimizer.state_dict()),
            'train_args': to_cpu(vars(self.args)),
            'train_metrics': {**self.metrics, 'fitness': self.fitness},
            'train_results': self.read_results_csv(),
            'date': datetime.now().isoformat(),
            'version': ultralytics.__version__,
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        data = buffer.getvalue()
        atomic_checkpoint(self.last, data)
        if self.best_fitness == self.fitness:
            atomic_checkpoint(self.best, data)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            atomic_checkpoint(self.wdir / f'epoch{self.epoch}.pt', data)

    def final_eval(self):
        # Reload the actual saved checkpoint and evaluate on CPU. Keep the
        # optimizer/epoch metadata so the checkpoint remains resumable.
        path = self.best if self.best.exists() else self.last
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        args = copy(self.args)
        args.device = 'cpu'
        args.half = False
        args.plots = False
        validator_type = PoseValidator if self.args.task == 'pose' else DetectionValidator
        validator = validator_type(self.test_loader, save_dir=self.save_dir, args=args)
        self.metrics = validator(model=checkpoint['ema'].float())
        print(f'Checkpoint reloaded and validated: {path}', flush=True)


class OpenCLPoseTrainer(OpenCLTrainerMixin, PoseTrainer):
    pass


class OpenCLDetectionTrainer(OpenCLTrainerMixin, DetectionTrainer):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model')
    parser.add_argument('--task', choices=('detect', 'pose'), default='detect')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--imgsz', type=int, default=128)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--name', default='pose-opencl')
    parser.add_argument('--resume')
    parser.add_argument('--project', default='runs/opencl')
    parser.add_argument('--device', default='ocl:0')
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--probe-steps', type=int, default=0)
    parser.add_argument('--warmup-steps', type=int, default=4)
    args = parser.parse_args()
    if args.probe_steps < 0 or args.warmup_steps < 1:
        parser.error('probe steps must be >=0 and warmup steps >=1')
    if args.batch < 1 or args.epochs < 1 or args.workers < 0:
        parser.error('batch/epochs must be positive and workers must be >=0')
    if torch.device(args.device).type != 'ocl':
        parser.error('this adapter requires an ocl device')
    if args.probe_steps and args.resume:
        parser.error('bounded probes load --model weights; do not combine with --resume')
    torch.backends.mkldnn.enabled = False
    torch.backends.nnpack.set_flags(False)  # Phenom cannot execute NNPACK kernels.
    torch.set_num_threads(2)
    if args.workers:
        torch.multiprocessing.set_start_method('spawn', force=True)
    model = args.model or ('yolov8n.pt' if args.task == 'detect' else 'yolov8n-pose.pt')
    overrides = dict(model=model, task=args.task, data=args.data, epochs=args.epochs, patience=args.patience,
                     imgsz=args.imgsz, batch=args.batch,
                     nbs=args.batch,
                     device=torch.device(args.device), amp=False, half=False,
                     workers=args.workers, cache=False, plots=False, deterministic=False,
                     optimizer='SGD', lr0=args.lr0, momentum=0.9, warmup_epochs=0,
                     mosaic=0, mixup=0, copy_paste=0,
                     project=args.project, name=args.name,
                     exist_ok=True, save=not bool(args.probe_steps), save_period=1)
    if args.resume:
        overrides['resume'] = args.resume
    trainer = OpenCLPoseTrainer if args.task == 'pose' else OpenCLDetectionTrainer
    instance = trainer(overrides=overrides)
    instance.args.device = str(instance.device)
    from training_telemetry import attach_training_telemetry
    attach_training_telemetry(instance)
    if not args.resume and not args.probe_steps:
        instance.add_callback('on_pretrain_routine_end', save_initial_checkpoint)
    class ProbeComplete(Exception):
        pass
    if args.probe_steps:
        import json
        import time
        probe = {'step': 0, 'start': None}
        def require_full_batches(t):
            full_batches = len(t.train_loader.dataset) // args.batch
            if full_batches < args.warmup_steps + args.probe_steps:
                raise ValueError('Dataset must supply enough full batches in one epoch for the probe window')
        instance.add_callback('on_train_start', require_full_batches)
        def batch_end(t):
            probe['step'] += 1
            if probe['step'] == args.warmup_steps:
                torch.ocl.synchronize(t.device)
                probe['start'] = time.perf_counter()
            if probe['step'] == args.warmup_steps + args.probe_steps:
                torch.ocl.synchronize(t.device)
                seconds = time.perf_counter() - probe['start']
                row = dict(batch=args.batch, measured_steps=args.probe_steps,
                           images_per_second=args.batch * args.probe_steps / seconds,
                           peak_reserved_bytes=torch.ocl.max_memory_reserved(t.device))
                (Path(t.save_dir) / 'benchmark.json').write_text(json.dumps(row, indent=2) + '\n')
                print('BENCHMARK', json.dumps(row), flush=True)
                raise ProbeComplete()
        instance.add_callback('on_train_batch_end', batch_end)
    try:
        instance.train()
    except ProbeComplete:
        print('PASS bounded throughput probe', flush=True)


if __name__ == '__main__':
    main()
