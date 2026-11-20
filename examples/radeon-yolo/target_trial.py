"""Fuse target gathers and one-hot score writes for the YOLO26n probe."""
import os
from types import MethodType
import torch


def install(model):
    from ultralytics.utils.tal import TaskAlignedAssigner
    for branch in (model.criterion.one2many,model.criterion.one2one):
        assigner=branch.assigner
        if type(assigner) is not TaskAlignedAssigner:
            continue

        def wrap(original):
            @torch.no_grad()
            def targets(self, labels, boxes, indices, foreground):
                if (os.environ.get('PROFILE_FUSED_TARGETS') != '1' or
                        labels.device.type != 'ocl' or labels.dtype != torch.float32 or
                        boxes.dtype != torch.float32 or self.num_classes != 5 or
                        labels.shape[1] == 0):
                    return original(labels,boxes,indices,foreground)
                return torch.ops.pytorch_ocl.tal_targets(labels,boxes,indices,foreground > 0,self.num_classes)
            return targets

        assigner.get_targets=MethodType(wrap(assigner.get_targets),assigner)
