"""Opt-in Conv-block training rewrite for OpenCL FP32 BN followed by SiLU."""
import os
from types import MethodType
import torch
from torch.autograd.function import once_differentiable


class TrainingBNSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,running_mean,running_var,momentum,eps):
        result,mean,var,scale,offset=torch.ops.pytorch_ocl.bn_silu_training(
            x,weight,bias,running_mean,running_var,momentum,eps)
        ctx.save_for_backward(x,weight,mean,var,scale,offset)
        ctx.eps=eps
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,weight,mean,var,scale,offset=ctx.saved_tensors
        normalized_grad=torch.ops.pytorch_ocl.bn_silu_recompute_grad(x,grad,scale,offset)
        dx,dw,db=torch.ops.aten.native_batch_norm_backward(
            normalized_grad,x,weight,None,None,mean,var,True,ctx.eps,list(ctx.needs_input_grad[:3]))
        return dx,dw,db,None,None,None,None


def install(model):
    from ultralytics.nn.modules.conv import Conv

    def wrap(original):
        def forward(self,x):
            norm=self.bn
            if (os.environ.get('PROFILE_BN_SILU')!='1' or not self.training or not norm.training or
                x.device.type!='ocl' or x.dtype!=torch.float32 or x.numel()==0 or
                not norm.affine or not norm.track_running_stats or norm.momentum is None):
                return original(x)
            value=self.conv(x)
            norm.num_batches_tracked.add_(1)
            return TrainingBNSiLU.apply(value,norm.weight,norm.bias,norm.running_mean,norm.running_var,
                                        norm.momentum,norm.eps)
        return forward

    count=0
    for module in model.modules():
        if (isinstance(module,Conv) and type(module).forward is Conv.forward and
            type(getattr(module,'bn',None)) is torch.nn.BatchNorm2d and
            type(getattr(module,'act',None)) is torch.nn.SiLU):
            module.forward=MethodType(wrap(module.forward),module)
            count+=1
    return count
