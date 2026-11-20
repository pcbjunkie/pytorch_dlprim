"""Try the existing OpenCL TAL metrics kernel on both YOLO26 loss branches."""
import inspect
import os
from types import MethodType
import torch

def install(criterion):
    from ultralytics.utils.tal import TaskAlignedAssigner
    for branch in (criterion.one2many,criterion.one2one):
        assigner=branch.assigner
        if type(assigner) is not TaskAlignedAssigner:
            continue
        original=assigner.get_box_metrics
        if not any(p.kind==p.VAR_KEYWORD for p in inspect.signature(original).parameters.values()):
            continue
        def wrap(original):
            @torch.no_grad()
            def metrics(self,scores,boxes,labels,gt,mask,**extra):
                if (os.environ.get('PROFILE_FUSED_METRICS')=='1' and not extra and
                    scores.device.type=='ocl' and all(t.dtype==torch.float32 for t in (scores,boxes,labels,gt))):
                    return torch.ops.pytorch_ocl.tal_box_metrics(scores,boxes,labels,gt,mask.bool(),self.alpha,self.beta)
                return original(scores,boxes,labels,gt,mask,**extra)
            return metrics
        assigner.get_box_metrics=MethodType(wrap(original),assigner)
