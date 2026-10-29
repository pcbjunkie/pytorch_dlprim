"""CPU-reference input gradients for power-of-two and fallback coordinate division.
Run with the r600 compatibility profile and candidate backend on PYTHONPATH.
"""
import torch
import pytorch_ocl
from torch.nn import functional as F

torch.set_num_threads(2)
torch.backends.mkldnn.enabled = False
torch.backends.nnpack.set_flags(False)
torch.manual_seed(2810)
for batch,ci,co,height,width,kernel,stride,groups,dilation in [
    (8,64,128,32,32,1,1,1,1),
    (8,64,64,32,32,1,1,1,1),
    (8,32,32,64,64,3,1,1,1),
    (8,128,256,16,16,3,2,1,1),
    (2,7,13,9,11,3,1,1,1),
    (2,4,8,9,11,3,2,4,1),
    (2,4,4,11,13,3,1,4,2),
]:
    x=torch.randn(batch,ci,height,width,requires_grad=True)
    w=torch.randn(co,ci//groups,kernel,kernel)
    padding=(kernel//2)*dilation
    y=F.conv2d(x,w,stride=stride,padding=padding,groups=groups,dilation=dilation)
    dy=torch.randn_like(y)
    y.backward(dy)
    gx=x.detach().to('ocl:0').requires_grad_()
    gw=w.to('ocl:0')
    for repeat in range(3):
        gx.grad=None
        F.conv2d(gx,gw,stride=stride,padding=padding,groups=groups,dilation=dilation).backward(dy.to('ocl:0'))
        torch.testing.assert_close(gx.grad.cpu(),x.grad,rtol=2e-3,atol=5e-4)
    print('PASS',batch,ci,co,height,width,kernel,stride,groups,dilation,flush=True)
