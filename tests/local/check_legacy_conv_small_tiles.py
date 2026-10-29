import os
os.environ["DLPRIM_CONV_GATHER"]="1"
"""CPU-reference convolution input gradients, plus complete supported conv autograd."""
assert os.environ.get('DLPRIM_GEMM_SMALL_TILES') == '1'
import copy,torch,pytorch_ocl
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;torch.backends.nnpack.set_flags(False);torch.manual_seed(6979)
cases=[(2,3,4,(3,3),(1,1),(1,1),(1,1),1,(9,11)),(2,4,6,(3,3),(2,2),(1,1),(1,1),2,(10,12)),(1,4,4,(3,3),(1,1),(1,1),(1,1),4,(11,13)),(1,4,8,(3,3),(1,1),(1,1),(1,1),4,(9,11)),(1,2,3,(1,1),(1,1),(0,0),(1,1),1,(7,9)),(1,3,5,(3,2),(2,1),(2,0),(2,1),1,(11,13)),(1,16,32,(3,3),(2,2),(1,1),(1,1),1,(32,32)),(1,4,4,(5,5),(1,1),(2,2),(1,1),4,(8,8))]
for n,ci,co,k,s,p,d,groups,hw in cases:
 x=torch.randn(n,ci,*hw,requires_grad=True);w=torch.randn(co,ci//groups,*k,requires_grad=True);y=F.conv2d(x,w,stride=s,padding=p,dilation=d,groups=groups);dy=torch.randn_like(y);y.backward(dy)
 actual=torch.ops.aten.convolution_backward_overrideable.default(dy.to('ocl:0'),x.detach().to('ocl:0'),w.detach().to('ocl:0'),s,p,d,False,[0,0],groups,[True,False,False])[0].cpu()
 torch.testing.assert_close(actual,x.grad,rtol=2e-3,atol=2e-4)
 print('PASS input gradient',ci,co,'kernel',k,'stride',s,'dilation',d,'groups',groups,hw,'maxerror',float((actual-x.grad).abs().max()),flush=True)
for ci,co,k,groups in [(3,4,3,1),(4,4,3,4),(4,6,3,2)]:
 ref=torch.nn.Conv2d(ci,co,k,padding=k//2,groups=groups);net=copy.deepcopy(ref).to('ocl:0');x=torch.randn(2,ci,9,11,requires_grad=True);g=x.detach().to('ocl:0').requires_grad_();a=ref(x);b=net(g);torch.testing.assert_close(b.cpu(),a,rtol=3e-4,atol=3e-5);dy=torch.randn_like(a);a.backward(dy);b.backward(dy.to('ocl:0'))
 for name,got,expected in [('input',g.grad,x.grad),('weight',net.weight.grad,ref.weight.grad),('bias',net.bias.grad,ref.bias.grad)]:torch.testing.assert_close(got.cpu(),expected,rtol=2e-3,atol=2e-4)
 print('PASS full convolution autograd',ci,co,k,groups,flush=True)

# Exact filter-gradient configuration that stalled the real YOLO probe, plus
# odd channel/spatial dimensions to exercise padded tile boundaries.
for ci,co,stride,hw in [(128,256,2,(16,16)),(7,13,1,(9,11))]:
 ref=torch.nn.Conv2d(ci,co,3,stride=stride,padding=1)
 net=copy.deepcopy(ref).to('ocl:0')
 x=torch.randn(2,ci,*hw,requires_grad=True)
 g=x.detach().to('ocl:0').requires_grad_()
 a=ref(x);b=net(g)
 torch.testing.assert_close(b.cpu(),a,rtol=5e-4,atol=1e-4)
 dy=torch.randn_like(a);a.backward(dy);b.backward(dy.to('ocl:0'))
 for name,got,expected in [('input',g.grad,x.grad),('weight',net.weight.grad,ref.weight.grad),('bias',net.bias.grad,ref.bias.grad)]:
  torch.testing.assert_close(got.cpu(),expected,rtol=2e-3,atol=2e-4)
 print('PASS full convolution exact/fringe',ci,co,'stride',stride,hw,flush=True)
for m,n,k in [(1,2,3),(7,13,11),(33,17,65),(256,257,128)]:
 a=torch.randn(m,k);b=torch.randn(k,n)
 got=torch.mm(a.to('ocl:0'),b.to('ocl:0')).cpu()
 torch.testing.assert_close(got,a@b,rtol=1e-3,atol=1e-4)
 print('PASS mm tile edges',m,n,k,flush=True)
