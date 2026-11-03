"""CPU-referenced max-pool gather gradients, including YOLO's repeated SPPF pool."""
import torch,pytorch_ocl
from torch.nn import functional as F
torch.set_num_threads(2);torch.manual_seed(2026)
cases=[((2,3,9,11),(3,3),(1,1),(1,1),False),((1,4,17,19),(5,5),(1,1),(2,2),False),((1,2,11,13),(3,5),(2,3),(1,2),True),((1,2,9,11),(2,2),(3,3),(0,0),False),((1,1,7,8),(9,9),(1,1),(4,4),False),((1,1,7,8),(13,13),(1,1),(6,6),False)]
for shape,k,s,p,ceil in cases:
 for tied in (False,True):
  x=(torch.ones(shape) if tied else torch.randn(shape)).requires_grad_();g=x.detach().to('ocl:0').requires_grad_()
  y=F.max_pool2d(x,k,s,p,ceil_mode=ceil);z=F.max_pool2d(g,k,s,p,ceil_mode=ceil)
  torch.testing.assert_close(z.cpu(),y,rtol=0,atol=0)
  dy=torch.randn_like(y);y.backward(dy);z.backward(dy.to('ocl:0'))
  torch.testing.assert_close(g.grad.cpu(),x.grad,rtol=3e-4,atol=3e-5)
  print('PASS',shape,'kernel',k,'stride',s,'padding',p,'ceil',ceil,'ties',tied,flush=True)
x=torch.randn(1,4,16,16,requires_grad=True);g=x.detach().to('ocl:0').requires_grad_()
def sppf(v):
 a=F.max_pool2d(v,5,1,2);b=F.max_pool2d(a,5,1,2);c=F.max_pool2d(b,5,1,2);return torch.cat((v,a,b,c),1)
y=sppf(x);z=sppf(g);dy=torch.randn_like(y);y.backward(dy);z.backward(dy.to('ocl:0'));torch.testing.assert_close(z.cpu(),y,rtol=0,atol=0);torch.testing.assert_close(g.grad.cpu(),x.grad,rtol=3e-4,atol=3e-5)
print('PASS three-stage SPPF output and accumulated input gradients',flush=True)
