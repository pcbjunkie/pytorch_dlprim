"""Ultralytics EMA with a fused OpenCL FP32 update and native fallback."""
import torch
from ultralytics.utils.torch_utils import ModelEMA, de_parallel


class OpenCLModelEMA(ModelEMA):
    @torch.no_grad()
    def update(self, model):
        if not self.enabled:
            return
        # Build the same floating-point state list as the upstream EMA. Resolve
        # it anew so moves, replaced buffers, and resumed state cannot go stale.
        current = de_parallel(model).state_dict()
        averages = []
        values = []
        for name, average in self.ema.state_dict().items():
            if average.dtype.is_floating_point:
                averages.append(average)
                values.append(current[name].detach())
        eligible = (averages and hasattr(torch.ops.dlprim, 'ema_update_') and
                    all(a.device.type == 'ocl' and a.device == v.device and
                        a.dtype == v.dtype == torch.float32 and
                        a.is_contiguous() and v.is_contiguous()
                        for a,v in zip(averages,values)))
        if not eligible:
            return super().update(model)
        self.updates += 1
        torch.ops.dlprim.ema_update_(averages, values, self.decay(self.updates))
