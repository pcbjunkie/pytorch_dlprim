"""CPU-reference atomic-free convolution filter reductions, including split tails."""
import torch,pytorch_ocl
from torch.nn import functional as F
import os
assert os.environ.get('DLPRIM_FILTER_SPLIT_K')=='1'
torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;torch.backends.nnpack.set_flags(False);torch.manual_seed(2811)
cases=[(2,3,16,64,64,3,2,1,1),(8,3,16,256,256,3,2,1,1),(2,16,16,64,64,3,1,1,1),(129,7,13,4,4,1,1,1,1),(2,8,12,64,64,3,1,2,1),(2,4,4,64,64,3,1,4,2),(7,3,16,64,64,3,2,1,1)]
if os.environ.get('SPLIT_CASE'):cases=[cases[int(os.environ['SPLIT_CASE'])]]
for bn,ci,co,h,w,k,st,g,dil in cases:
 x=torch.randn(bn,ci,h,w);weight=torch.randn(co,ci//g,k,k,requires_grad=True);pad=(k//2)*dil
 y=F.conv2d(x,weight,stride=st,padding=pad,groups=g,dilation=dil);dy=torch.randn_like(y);y.backward(dy)
 gx=x.to('ocl:0');gw=weight.detach().to('ocl:0').requires_grad_();gdy=dy.to('ocl:0')
 for repeat in range(3):
  gw.grad=None
  F.conv2d(gx,gw,stride=st,padding=pad,groups=g,dilation=dil).backward(gdy)
  actual=gw.grad.cpu()
  if not torch.allclose(actual,weight.grad,rtol=3e-3,atol=3e-3):
   os.environ['DLPRIM_FILTER_SPLIT_K']='0';gw.grad=None
   F.conv2d(gx,gw,stride=st,padding=pad,groups=g,dilation=dil).backward(gdy)
   control=gw.grad.cpu()
   print('CONTROL maximum CPU error',float((control-weight.grad).abs().max()),'CANDIDATE',float((actual-weight.grad).abs().max()),flush=True)
   print('first CPU',weight.grad.flatten()[:8],'control',control.flatten()[:8],'candidate',actual.flatten()[:8],flush=True)
   os.environ['DLPRIM_FILTER_SPLIT_K']='1'
  torch.testing.assert_close(actual,weight.grad,rtol=3e-3,atol=3e-3)
 print('PASS',bn,ci,co,h,w,k,st,g,dil,'maxerror',float((actual-weight.grad).abs().max()),flush=True)
 # Two backward calls must accumulate rather than overwrite the leaf gradient.
 F.conv2d(gx,gw,stride=st,padding=pad,groups=g,dilation=dil).backward(gdy)
 torch.testing.assert_close(gw.grad.cpu(),2*weight.grad,rtol=3e-3,atol=6e-3)
print('PASS split filter regression',flush=True)
