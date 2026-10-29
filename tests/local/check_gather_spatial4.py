"""CPU-reference input gradients for spatial 3x3, small 1x1 and fallback paths.
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
    (24,16,32,128,128,3,2,1,1),
    (24,128,256,16,16,3,2,1,1),
    (8,32,32,64,64,3,1,1,1),
    (2,8,16,17,40,3,2,1,1),
    (2,8,16,9,36,3,1,1,1),
    (2,8,16,9,16,3,1,1,1),
    (2,8,16,17,16,3,2,1,1),
    (2,8,16,9,35,3,1,1,1),
    (2,8,16,17,36,3,2,1,1),
    (2,8,16,9,32,3,1,2,1),
    (8,64,128,32,32,1,1,1,1),
    (24,32,64,128,128,1,1,1,1),
    (24,64,128,64,64,1,1,1,1),
    (2,7,13,9,36,1,1,1,1),
    (2,8,16,9,36,1,1,4,1),
    (2,4,4,11,36,1,1,4,1),
    (2,7,13,9,35,1,1,1,1),
    (2,64,32,16,16,1,1,1,1),
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
print('PASS spatial4, small1x1 and scalar fallback input-gradient regression',flush=True)
