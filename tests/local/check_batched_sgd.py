"""Check native torch.optim.SGD behavior with OpenCL foreach kernels."""
import copy
import warnings
import torch,pytorch_ocl
from torch.utils._pytree import tree_map

torch.set_num_threads(2);torch.manual_seed(53)

def close(a,b):torch.testing.assert_close(a.cpu(),b,rtol=3e-6,atol=3e-7,equal_nan=True)
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 for count in [1,9,16,17,33]:
  a=[torch.randn(i%7+1,11) for i in range(count)];b=[torch.randn_like(x) for x in a]
  ga=[x.to('ocl:0') for x in a];gb=[x.to('ocl:0') for x in b]
  actual=torch._foreach_add(ga,gb,alpha=.3)
  for x,y in zip(actual,torch._foreach_add(a,b,alpha=.3)):close(x,y)
  versions=[x._version for x in ga]
  torch._foreach_add_(ga,gb,alpha=-.9);torch._foreach_add_(a,b,alpha=-.9)
  assert all(x._version>v for x,v in zip(ga,versions))
  torch._foreach_mul_(ga,.7);torch._foreach_mul_(a,.7)
  torch._foreach_mul_(ga,torch.tensor(.4,device='ocl:0'));torch._foreach_mul_(a,torch.tensor(.4))
  for x,y in zip(ga,a):close(x,y)
 # Native fallback handles strided/dtype/broadcast and aliased inputs.
 for dtype in [torch.float32,torch.int64]:
  a=torch.arange(12).reshape(3,4).t().to(dtype);b=torch.ones_like(a)
  ga=a.to('ocl:0');gb=b.to('ocl:0')
  torch._foreach_add_([ga],[gb],alpha=2);torch._foreach_add_([a],[b],alpha=2);close(ga,a)
 x=torch.ones(7,device='ocl:0');torch._foreach_mul_([x,x],2);close(x,torch.full((7,),4.))
 x=torch.arange(5.,device='ocl:0');y=torch.ones(5,device='ocl:0')
 expected_x=x.cpu()+y.cpu();expected_y=y.cpu()+expected_x
 torch._foreach_add_([x,y],[y,x]);close(x,expected_x);close(y,expected_y)
 # General foreach gradients must still work through the native path.
 a=torch.randn(5,requires_grad=True);b=torch.randn(5,requires_grad=True)
 ga=a.detach().to('ocl:0').requires_grad_();gb=b.detach().to('ocl:0').requires_grad_()
 torch._foreach_add([ga],[gb],alpha=.3)[0].sum().backward()
 torch._foreach_add([a],[b],alpha=.3)[0].sum().backward();close(ga.grad,a.grad);close(gb.grad,b.grad)
 configs=[dict(momentum=0.,weight_decay=0.),dict(momentum=.9,weight_decay=.0005),
          dict(momentum=.9,weight_decay=.0005,nesterov=True),
          dict(momentum=.8,dampening=.3,weight_decay=.02),
          dict(momentum=.9,weight_decay=.001,nesterov=True,maximize=True)]
 for config in configs:
  cpu=[torch.nn.Parameter(torch.randn(3,7)),torch.nn.Parameter(torch.randn(19)),torch.nn.Parameter(torch.randn(2,5))]
  gpu=[torch.nn.Parameter(x.detach().to('ocl:0')) for x in cpu]
  oc=torch.optim.SGD([{'params':cpu[:2]},{'params':cpu[2:],'weight_decay':0.}],lr=.003,foreach=True,**config)
  og=torch.optim.SGD([{'params':gpu[:2]},{'params':gpu[2:],'weight_decay':0.}],lr=.003,foreach=True,**config)
  for step in range(20):
   for i,(x,y) in enumerate(zip(cpu,gpu)):
    x.grad=torch.randn_like(x) if not (i==1 and step%4==0) else None
    y.grad=x.grad.to('ocl:0') if x.grad is not None else None
   if step==10:
    for opt in (oc,og):
     for group in opt.param_groups:group['lr']=.001
   oc.step();og.step()
   for x,y in zip(cpu,gpu):
    close(y,x.detach())
    if x.grad is not None:close(y.grad,x.grad)
    for key,value in oc.state[x].items():close(og.state[y][key],value)
  # Round-trip standard optimizer state onto new device parameters.
  state=tree_map(lambda x:x.detach().cpu() if isinstance(x,torch.Tensor) else x,og.state_dict())
  restored=[torch.nn.Parameter(x.detach().clone()) for x in gpu]
  resume=torch.optim.SGD([{'params':restored[:2]},{'params':restored[2:]}],lr=.9,foreach=True,**config)
  resume.load_state_dict(state)
  for x,y in zip(restored,gpu):
   for key,value in og.state[y].items():close(resume.state[x][key],value.cpu())
  for x,y in zip(restored,gpu):
   grad=torch.randn(x.shape);x.grad=grad.to('ocl:0');y.grad=grad.to('ocl:0')
  resume.step();og.step()
  for x,y in zip(restored,gpu):close(x,y.detach().cpu())
 unsupported=[str(w.message) for w in caught if 'not currently supported' in str(w.message)]
 assert not unsupported,unsupported
print('PASS: foreach add/mul, versions/alias/layout/autograd fallback; native SGD momentum/Nesterov/decay/dampening/maximize/missing-grad/LR/state resume')
