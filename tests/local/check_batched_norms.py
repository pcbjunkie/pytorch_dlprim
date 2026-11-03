"""Reference tests for batched foreach norms and packed scalar stacking."""
import warnings
import torch,pytorch_ocl

torch.set_num_threads(2)
torch.manual_seed(31)
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 for count in [1,9,16,17,33,202]:
  cpu=[torch.randn((i%7+1)*19) for i in range(count)]
  if count>1:cpu[1]=torch.empty(0)
  gpu=[x.to('ocl:0') for x in cpu]
  expected=torch._foreach_norm(cpu,2)
  actual=torch._foreach_norm(gpu,2)
  torch.testing.assert_close(torch.stack(actual).cpu(),torch.stack(expected),rtol=2e-5,atol=2e-6)
  for x,y in zip(gpu,cpu):torch.testing.assert_close(x.cpu(),y,rtol=0,atol=0)
  joined=torch.stack(actual);before=actual[0].cpu().clone();joined[0]=0
  torch.testing.assert_close(actual[0].cpu(),before,rtol=0,atol=0)
 # Long reductions, storage offsets, nonfinite handling and fallback variants.
 cpu=[torch.full((2_000_000,),.001),torch.randn(700_003)[3:],torch.tensor([1.,float('inf')]),
      torch.tensor([float('inf'),float('nan')]),torch.tensor([1e20,1e20]),torch.zeros(31),torch.tensor(3.)]
 actual=torch._foreach_norm([x.to('ocl:0') for x in cpu],2)
 expected=list(torch._foreach_norm(cpu,2))
 # The old non-AVX CPU FP32 reduction loses precision on two million
 # repeated .001 values; compare that adversarial case to FP64 instead.
 accurate=torch.linalg.vector_norm(cpu[0].double()).float()
 print('Long uniform norm: CPU FP32',expected[0].item(),'GPU',actual[0].cpu().item(),'CPU FP64 reference',accurate.item(),flush=True)
 expected[0]=accurate
 torch.testing.assert_close(torch.stack(actual).cpu(),torch.stack(expected),rtol=2e-5,atol=2e-6,equal_nan=True)
 for p in [0,1,3,-2,float('inf'),-float('inf')]:
  cpu=[torch.randn(4,7).t(),torch.randn(19)]
  gpu=[x.to('ocl:0') for x in cpu]
  torch.testing.assert_close(torch.stack(torch._foreach_norm(gpu,p)).cpu(),torch.stack(torch._foreach_norm(cpu,p)),rtol=2e-5,atol=2e-6)
 cpu=[torch.randn(31),torch.randn(7)]
 values=torch._foreach_norm([x.to('ocl:0') for x in cpu],2,dtype=torch.float64)
 torch.testing.assert_close(torch.stack(values).cpu(),torch.stack(torch._foreach_norm(cpu,2,dtype=torch.float64)),rtol=2e-5,atol=2e-6)
 # Native clipping still combines norms and scales grads through Torch.
 for nonfinite in [False,True]:
  params=[torch.nn.Parameter(torch.randn(i%5+1,13)) for i in range(202)]
  gpu=[torch.nn.Parameter(x.detach().to('ocl:0')) for x in params]
  for x,y in zip(params,gpu):x.grad=torch.randn_like(x);y.grad=x.grad.to('ocl:0')
  if nonfinite:
   params[0].grad.flatten()[0]=float('inf');gpu[0].grad.flatten()[0]=float('inf')
  c=torch.nn.utils.clip_grad_norm_(params,.7)
  g=torch.nn.utils.clip_grad_norm_(gpu,.7)
  torch.testing.assert_close(g.cpu(),c,rtol=2e-5,atol=2e-6,equal_nan=True)
  for x,y in zip(params,gpu):torch.testing.assert_close(y.grad.cpu(),x.grad,rtol=2e-5,atol=2e-6,equal_nan=True)
 # General stack must retain native semantics and autograd.
 for shape in [(),(3,),(2,3)]:
  for dim in range(-len(shape)-1,len(shape)+1):
   cpu=[torch.randn(shape,requires_grad=True) for _ in range(3)]
   gpu=[x.detach().to('ocl:0').requires_grad_() for x in cpu]
   a=torch.stack(gpu,dim);b=torch.stack(cpu,dim)
   torch.testing.assert_close(a.cpu(),b);a.sum().backward();b.sum().backward()
   for x,y in zip(gpu,cpu):torch.testing.assert_close(x.grad.cpu(),y.grad)
 x=torch.randn(17,requires_grad=True);g=x.detach().to('ocl:0').requires_grad_()
 a=torch.stack(g.unbind());b=torch.stack(x.unbind());a.sum().backward();b.sum().backward()
 torch.testing.assert_close(g.grad.cpu(),x.grad)
 # Differentiable foreach retains independent output semantics.
 cpu=[torch.randn(4,7,requires_grad=True),torch.randn(13,requires_grad=True)]
 gpu=[x.detach().to('ocl:0').requires_grad_() for x in cpu]
 a=torch.stack(torch._foreach_norm(gpu,2)).sum();b=torch.stack(torch._foreach_norm(cpu,2)).sum()
 a.backward();b.backward()
 for x,y in zip(gpu,cpu):torch.testing.assert_close(x.grad.cpu(),y.grad,rtol=2e-5,atol=2e-6)
 # Rejecting a nonfinite total remains the native clipping behavior.
 p=torch.nn.Parameter(torch.ones(3,device='ocl:0'));p.grad=torch.tensor([float('nan'),1,2],device='ocl:0')
 try:torch.nn.utils.clip_grad_norm_([p],1,error_if_nonfinite=True)
 except RuntimeError:pass
 else:raise AssertionError('nonfinite clipping did not raise')
 # The optional error_if_nonfinite guard already falls back for two scalar
 # boolean operations; normal trainer clipping does not use that guard.
 unsupported=[str(w.message) for w in caught if 'not currently supported' in str(w.message)
              and not any(name in str(w.message) for name in ['aten::isnan', 'aten::logical_or.out'])]
 assert not unsupported,unsupported
print('PASS: batched norms, packed stack, CPU clipping reference, fallback orders/dtypes, nonfinite/large inputs, stack gradients')
