"""Opt-in adapter for the DFL-free Ultralytics regression loss."""
import os
from types import MethodType

import torch
from torch.autograd.function import once_differentiable


class FusedCIoUL1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist, boxes, anchors, target, scores, norm, mask, extent, stride):
        iou, l1, dist_grad, box_grad = torch.ops.pytorch_ocl.bbox_loss_l1(
            dist, boxes, anchors, target, scores, norm, mask, extent, stride
        )
        ctx.save_for_backward(dist_grad, box_grad)
        return iou, l1

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_iou, grad_l1):
        dist_grad, box_grad = ctx.saved_tensors
        return (
            dist_grad * grad_l1 if ctx.needs_input_grad[0] else None,
            box_grad * grad_iou if ctx.needs_input_grad[1] else None,
            None, None, None, None, None, None, None,
        )


def install(model):
    from ultralytics.utils.loss import BboxLoss

    def wrap(original):
        def forward(self, dist, boxes, anchors, target, scores, normalizer, mask, imgsz, stride):
            args = dist, boxes, anchors, target, scores, normalizer, mask, imgsz, stride
            runtime_constants = anchors, target, scores, normalizer, imgsz, stride
            supported = (
                os.environ.get('PROFILE_V26_CIOU_L1') == '1'
                and dist.device.type == 'ocl' and dist.ndim == 3 and dist.shape[-1] == 4
                and 0 < dist.shape[1] <= 65535 and mask.dtype == torch.bool
                and all(t.device == dist.device and t.dtype == torch.float32
                        for t in (dist, boxes, *runtime_constants))
                and not any(t.requires_grad for t in runtime_constants)
                and stride.shape == (dist.shape[1], 1)
            )
            if not supported:
                return original(*args)
            return FusedCIoUL1.apply(*args)
        return forward

    for branch in (model.criterion.one2many, model.criterion.one2one):
        module = branch.bbox_loss
        if type(module) is BboxLoss and module.dfl_loss is None:
            module.forward = MethodType(wrap(module.forward), module)
