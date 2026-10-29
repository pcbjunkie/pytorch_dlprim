"""Fused SiLU backward vs CPU: tails, offsets, strided operands/out, autograd."""
import os,torch,pytorch_ocl
from torch.nn import functional as F
assert os.environ.get('DLPRIM_FUSED_SILU_BACKWARD')=='1'
torch.set_num_threads(2);torch.manual_seed(2813)
for shape in [(),(0,),(1,),(31,),(32,),(33,),(127,),(128,),(129,),(3,7,11),(16,64,32,32),(24,16,128,128)]:
 x=torch.randn(shape)*4;dy=torch.randn(shape)
 ref=torch.ops.aten.silu_backward.default(dy,x)
 got=torch.ops.aten.silu_backward.default(dy.to('ocl:0'),x.to('ocl:0')).cpu()
 torch.testing.assert_close(got,ref,rtol=3e-5,atol=3e-6)
 print('PASS shape',shape,flush=True)
x=torch.tensor([-float('inf'),-100.,-50.,-20.,-1.,-0.,0.,1.,20.,50.,100.,float('inf'),float('nan')]);dy=torch.ones_like(x)
ref=torch.ops.aten.silu_backward.default(dy,x);got=torch.ops.aten.silu_backward.default(dy.to('ocl:0'),x.to('ocl:0')).cpu()
torch.testing.assert_close(got,ref,rtol=3e-5,atol=3e-6,equal_nan=True);print('PASS finite extremes and NaN/Inf',flush=True)
base=torch.randn(19,23);grad=torch.randn(19,23);x=base[2:17:2,3:21:2];dy=grad[2:17:2,3:21:2]
gx=base.to('ocl:0')[2:17:2,3:21:2];gdy=grad.to('ocl:0')[2:17:2,3:21:2]
ref=torch.ops.aten.silu_backward.default(dy,x)
got=torch.ops.aten.silu_backward.default(gdy,gx).cpu();torch.testing.assert_close(got,ref,rtol=3e-5,atol=3e-6)
out=torch.full((19,23),123.,device='ocl:0');view=out[2:17:2,3:21:2]
torch.ops.aten.silu_backward.grad_input(gdy,gx,grad_input=view)
expected=torch.full((19,23),123.);expected[2:17:2,3:21:2]=ref
torch.testing.assert_close(out.cpu(),expected,rtol=3e-5,atol=3e-6);print('PASS strided inputs/output, storage offsets and neighbors',flush=True)
for inplace in [False,True]:
 x=torch.randn(2,7,9,11,requires_grad=True);dy=torch.randn_like(x)
 F.silu(x+0,inplace=inplace).backward(dy)
 gx=x.detach().to('ocl:0').requires_grad_();F.silu(gx+0,inplace=inplace).backward(dy.to('ocl:0'))
 torch.testing.assert_close(gx.grad.cpu(),x.grad,rtol=3e-5,atol=3e-6)
 print('PASS autograd inplace',inplace,flush=True)
# Unsupported fused shape case keeps the original composed implementation.
x=torch.randn(2,3);dy=torch.randn(1,3)
ref=torch.ops.aten.silu_backward.default(dy,x);got=torch.ops.aten.silu_backward.default(dy.to('ocl:0'),x.to('ocl:0')).cpu()
torch.testing.assert_close(got,ref,rtol=3e-5,atol=3e-6);print('PASS broadcast fallback',flush=True)
x=torch.randn(7,requires_grad=True);dy=torch.randn(7,requires_grad=True)
sigmoid=x.sigmoid();y=dy*sigmoid*(1+x*(1-sigmoid))
reference=torch.autograd.grad(y.sum(),(x,dy))
gx=x.detach().to('ocl:0').requires_grad_();gdy=dy.detach().to('ocl:0').requires_grad_()
gy=torch.ops.aten.silu_backward.default(gdy,gx)
actual=torch.autograd.grad(gy.sum(),(gx,gdy))
for a,b in zip(actual,reference):torch.testing.assert_close(a.cpu(),b,rtol=3e-5,atol=3e-6)
print('PASS higher-order autograd fallback',flush=True)
print('PASS fused SiLU backward regression',flush=True)
