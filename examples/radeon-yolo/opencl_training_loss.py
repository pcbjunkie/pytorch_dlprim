"""Opt-in YOLO TaskAlignedAssigner optimizations; original Ultralytics stays intact."""
# Top-k batching adapted from Ultralytics8.3.0 utils/tal.py (AGPL-3.0).
import os,types,inspect
import torch

def optimize_assigner(assigner):
    from ultralytics.utils.tal import TaskAlignedAssigner
    # Rotated/subclass assigners can use different overlap definitions.
    if type(assigner) is not TaskAlignedAssigner:return
    if 'largest' not in inspect.signature(type(assigner).select_topk_candidates).parameters:return
    if not any(os.environ.get(k)=='1' for k in ('DLPRIM_TAL_FUSED_METRICS','DLPRIM_TAL_VECTOR_SCATTER')):return
    if getattr(assigner,'_opencl_loss_optimized',False):return
    if os.environ.get('DLPRIM_TAL_FUSED_METRICS')=='1':
        original=assigner.get_box_metrics
        @torch.no_grad()
        def metrics(self,scores,boxes,labels,gt,mask):
            if scores.device.type!='ocl' or any(t.dtype!=torch.float32 for t in (scores,boxes,labels,gt)):
                return original(scores,boxes,labels,gt,mask)
            return torch.ops.pytorch_ocl.tal_box_metrics(scores,boxes,labels,gt,mask.bool(),self.alpha,self.beta)
        assigner.get_box_metrics=types.MethodType(metrics,assigner)
    if os.environ.get('DLPRIM_TAL_VECTOR_SCATTER')=='1':
        original_topk=assigner.select_topk_candidates
        @torch.no_grad()
        def topk(self,metrics,largest=True,topk_mask=None):
            if metrics.device.type!='ocl':return original_topk(metrics,largest,topk_mask)
            values,indices=torch.topk(metrics,self.topk,dim=-1,largest=largest)
            if topk_mask is None:topk_mask=(values.max(-1,keepdim=True)[0]>self.eps).expand_as(indices)
            indices.masked_fill_(~topk_mask,0)
            counts=torch.zeros(metrics.shape,dtype=torch.int8,device=metrics.device)
            counts.scatter_add_(-1,indices,torch.ones_like(indices,dtype=torch.int8))
            counts.masked_fill_(counts>1,0)
            return counts.to(metrics.dtype)
        assigner.select_topk_candidates=types.MethodType(topk,assigner)
    assigner._opencl_loss_optimized=True

class _FusedBoxDFL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,dist,boxes,anchors,target,scores,normalizer,mask):
        li,ld,gd,gp=torch.ops.pytorch_ocl.bbox_loss16(dist,boxes,anchors,target,scores,normalizer,mask)
        ctx.save_for_backward(gd,gp,dist,boxes,anchors,target,scores,normalizer,mask)
        return li,ld
    @staticmethod
    def backward(ctx,grad_iou,grad_dfl):
        gd,gp,dist,boxes,anchors,target,scores,normalizer,mask=ctx.saved_tensors
        if torch.is_grad_enabled():
            # Separate downstream aliases make the local Jacobians independent
            # even when decoded boxes depend on the original distribution tensor.
            from ultralytics.utils.loss import BboxLoss
            d=dist.view_as(dist);b=boxes.view_as(boxes)
            li,ld=BboxLoss(16)(d,b,anchors,target,scores,normalizer,mask)
            inputs=[x for x,need in zip((d,b),ctx.needs_input_grad[:2]) if need]
            grads=iter(torch.autograd.grad((li,ld),inputs,(grad_iou,grad_dfl),create_graph=True))
            return tuple(next(grads) if need else None for need in ctx.needs_input_grad[:2])+(None,)*5
        return (gd*grad_dfl if ctx.needs_input_grad[0] else None,
                gp*grad_iou if ctx.needs_input_grad[1] else None,
                None,None,None,None,None)

def optimize_bbox_loss(module):
    from ultralytics.utils.loss import BboxLoss
    if os.environ.get('DLPRIM_FUSED_BOX_DFL')!='1' or type(module) is not BboxLoss:return
    if len(inspect.signature(type(module).forward).parameters)!=8:return
    if getattr(module,'_opencl_bbox_optimized',False):return
    original=module.forward
    def forward(self,dist,boxes,anchors,target,scores,normalizer,mask):
        supported=(dist.device.type=='ocl' and dist.dtype==torch.float32 and boxes.dtype==torch.float32 and dist.ndim==3 and dist.shape[-1]==64
            and dist.shape[1]<=65535 and self.dfl_loss is not None and self.dfl_loss.reg_max==16
            and all(x.dtype==torch.float32 and not x.requires_grad for x in (anchors,target,scores))
            and (not isinstance(normalizer,torch.Tensor) or not normalizer.requires_grad))
        if not supported:return original(dist,boxes,anchors,target,scores,normalizer,mask)
        norm=torch.as_tensor(normalizer,dtype=torch.float32,device=dist.device)
        return _FusedBoxDFL.apply(dist,boxes,anchors,target,scores,norm,mask)
    module.forward=types.MethodType(forward,module)
    module._opencl_bbox_optimized=True

def install(model):
    if not any(os.environ.get(k)=='1' for k in ('DLPRIM_TAL_FUSED_METRICS','DLPRIM_TAL_VECTOR_SCATTER','DLPRIM_FUSED_BOX_DFL')):return
    if not hasattr(model,'criterion'):model.criterion=model.init_criterion()
    assigner=getattr(model.criterion,'assigner',None)
    if assigner is not None:optimize_assigner(assigner)
    bbox=getattr(model.criterion,'bbox_loss',None)
    if bbox is not None:optimize_bbox_loss(bbox)
