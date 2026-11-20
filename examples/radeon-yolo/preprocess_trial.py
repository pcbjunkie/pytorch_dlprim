"""Share one target preparation across both loss branches for CPU-label batches."""
import copy
import os
from types import MethodType

import torch


def install(model):
    branches = (model.criterion.one2many, model.criterion.one2one)
    original_loss = model.loss
    host_branch = copy.copy(branches[0])
    host_branch.device = torch.device('cpu')

    def host_targets(fields, images):
        bidx, classes, boxes = fields
        labels = torch.cat((bidx.reshape(-1, 1), classes.reshape(-1, 1), boxes), dim=1)
        height, width = images.shape[-2:]
        scale = torch.tensor([width, height, width, height], dtype=images.dtype)
        return labels, host_branch.preprocess(labels, images.shape[0], scale)

    def stage_targets(batch):
        fields = [batch[name] for name in ('batch_idx', 'cls', 'bboxes')]
        images = batch['img']
        if (os.environ.get('PROFILE_V26_STAGE_TARGETS') != '1' or
                os.environ.get('PROFILE_SHARED_TARGETS') != '1' or images.device.type != 'ocl' or
                images.dtype != torch.float32 or any(n % 32 for n in images.shape[-2:]) or
                any(t.device.type != 'cpu' or t.requires_grad for t in fields)):
            return
        labels, targets = host_targets(fields, images)
        batch['_opencl_prepared_targets'] = targets.to(images.device)
        labels = labels.to(images.device)
        batch.update(batch_idx=labels[:, 0], cls=labels[:, 1:2], bboxes=labels[:, 2:])

    def shared_loss(self, batch, preds=None):
        fields = [batch[name] for name in ('batch_idx', 'cls', 'bboxes')]
        images = batch['img']
        if os.environ.get('PROFILE_SHARED_TARGETS') != '1' or preds is None:
            return original_loss(batch, preds=preds)
        targets = batch.get('_opencl_prepared_targets')
        if targets is None:
            if (any(t.device.type != 'cpu' or t.requires_grad for t in fields) or
                    images.device.type != 'ocl' or images.dtype != torch.float32 or
                    any(n % 32 for n in images.shape[-2:])):
                return original_loss(batch, preds=preds)
            _, targets = host_targets(fields, images)
            targets = targets.to(images.device)

        def prepared(unused_targets, batch_size, scale_tensor):
            return targets

        previous = [branch.preprocess for branch in branches]
        try:
            for branch in branches:
                branch.preprocess = prepared
            return original_loss(batch, preds=preds)
        finally:
            for branch, fn in zip(branches, previous):
                branch.preprocess = fn

    model.loss = MethodType(shared_loss, model)
    return stage_targets
