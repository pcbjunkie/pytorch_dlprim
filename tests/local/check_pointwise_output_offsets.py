"""Pointwise output views preserve their offsets and surrounding elements."""
import torch,pytorch_ocl
torch.set_num_threads(2)
checks=0
for dtype in [torch.float32,torch.int32,torch.uint8,torch.int8]:
 for offset in [0,1,2,7,63,64,65,127]:
  for length in [1,3,64,65]:
   for op in ['explicit_scalar','python_scalar','pow_out','add_out']:
    c=((torch.arange(offset+length+5)%9)+1).to(dtype);g=c.to('ocl:0')
    a=c[offset:offset+length];b=g[offset:offset+length]
    if op=='explicit_scalar':torch.ops.aten.mul_.Scalar(a,2);torch.ops.aten.mul_.Scalar(b,2)
    elif op=='python_scalar':a.mul_(2);b.mul_(2)
    elif op=='pow_out':
     if dtype!=torch.float32:continue
     torch.pow(a,2,out=a);torch.pow(b,2,out=b)
    else:torch.add(a,2,out=a);torch.add(b,2,out=b)
    torch.testing.assert_close(g.cpu(),c,rtol=0,atol=0,msg=lambda m:f'{dtype} offset{offset} length{length} {op}: {m}');checks+=1
 print('PASS offsets/neighbor preservation',dtype,flush=True)
# Scalars selected from a shared three-element loss tensor.
c=torch.tensor([1.5,2.,3.]);g=c.to('ocl:0')
for i,scale in enumerate([7.5,.5,1.5]):
 torch.ops.aten.mul_.Scalar(c[i],scale);torch.ops.aten.mul_.Scalar(g[i],scale)
torch.testing.assert_close(g.cpu(),c,rtol=0,atol=0);checks+=1
print('PASS',checks,'offset/neighbor CPU comparisons',flush=True)
