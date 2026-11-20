"""Explicit inference-only graph rewrites for already BN-fused YOLO Conv blocks."""
import types
import torch

def fuse_conv_silu(model):
    from ultralytics.nn.modules.conv import Conv, DWConv
    eligible=(Conv,DWConv)
    try:
        from yolox.models.network_blocks import BaseConv
        eligible+=(BaseConv,)
    except ImportError:pass
    names=[]
    def forward(self,x):
        c=self.conv
        return torch.ops.pytorch_ocl.convolution_silu(x,c.weight,c.bias,
                    list(c.stride),list(c.padding),list(c.dilation),c.groups)
    for name,module in model.named_modules():
        # Exclude Conv subclasses that implement additional branches.
        if type(module) in eligible and isinstance(module.act,torch.nn.SiLU):
            if hasattr(module,'bn'):raise ValueError('Batch normalization must be fused first')
            if module.training:raise ValueError('Fusion requires eval mode')
            module.forward=types.MethodType(forward,module)
            names.append(name)
    if not names:raise ValueError('No eligible Conv/SiLU blocks found')
    return names


def fuse_sppf(model):
    from ultralytics.nn.modules.block import SPPF
    names=[]
    def forward(self,x):
        y=self.cv2(torch.ops.pytorch_ocl.sppf_pool_cat(self.cv1(x)))
        return y+x if getattr(self,"add",False) else y
    for name,module in model.named_modules():
        if type(module) is SPPF:
            m=module.m
            if m.kernel_size!=5 or m.stride!=1 or m.padding!=2 or m.dilation!=1 or m.ceil_mode or m.return_indices:
                raise ValueError('SPPF requires ordinary 5x5/stride1/pad2 max pool')
            if getattr(module,'n',3)!=3:raise ValueError('SPPF fusion requires three pooling iterations')
            if module.training:raise ValueError('SPPF fusion requires eval mode')
            module.forward=types.MethodType(forward,module);names.append(name)
    try:
        from yolox.models.network_blocks import SPPBottleneck
    except ImportError:SPPBottleneck=None
    if SPPBottleneck is not None:
        def spp_forward(self,x):
            return self.conv2(torch.ops.pytorch_ocl.sppf_pool_cat(self.conv1(x)))
        for name,module in model.named_modules():
            if type(module) is SPPBottleneck:
                if [m.kernel_size for m in module.m]!=[5,9,13] or any(m.stride!=1 or m.padding!=m.kernel_size//2 or m.dilation!=1 or m.ceil_mode or m.return_indices for m in module.m):
                    raise ValueError('SPP rewrite requires ordinary parallel 5/9/13 pools')
                if module.training:raise ValueError('SPP rewrite requires eval mode')
                module.forward=types.MethodType(spp_forward,module);names.append(name)
    if not names:raise ValueError('No eligible SPP/SPPF blocks found')
    return names


def accept_precomputed_focus(model):
    from yolox.models.network_blocks import Focus
    names=[]
    def forward(self,x):return self.conv(x)
    for name,module in model.named_modules():
        if type(module) is Focus:
            if module.conv.conv.in_channels!=12 or module.training:
                raise ValueError('Expected eval-mode three-channel YOLOX Focus stem')
            module.forward=types.MethodType(forward,module);names.append(name)
    if len(names)!=1:raise ValueError('Expected exactly one Focus stem')
    return names
