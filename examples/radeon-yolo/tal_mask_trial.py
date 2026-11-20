"""Apply the existing top-k selection with a single device mask write."""
import os
from types import MethodType

import torch


def install(model):
    from ultralytics.utils.tal import TaskAlignedAssigner

    def wrap(original):
        @torch.no_grad()
        def select(self, metrics, topk_mask=None):
            if (os.environ.get('PROFILE_V26_TAL_MASK') != '1' or metrics.device.type != 'ocl'
                    or not 0 < self.topk <= min(32, metrics.shape[-1])):
                return original(metrics, topk_mask=topk_mask)
            values, indices = torch.topk(metrics, self.topk, dim=-1)
            if topk_mask is None:
                topk_mask = (values.max(-1, keepdim=True).values > self.eps).expand_as(indices)
            if topk_mask.dtype != torch.bool or topk_mask.shape != indices.shape:
                return original(metrics, topk_mask=topk_mask)
            return torch.ops.pytorch_ocl.tal_topk_mask(indices, topk_mask, metrics.shape[-1])
        return select

    for branch in (model.criterion.one2many, model.criterion.one2one):
        assigner = branch.assigner
        if type(assigner) is TaskAlignedAssigner:
            assigner.select_topk_candidates = MethodType(wrap(assigner.select_topk_candidates), assigner)
