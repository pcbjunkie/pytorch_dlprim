"""CPU-reference checks for GPU strided copies, offsets and untouched neighbors."""
import torch,pytorch_ocl
torch.set_num_threads(2)
count=0
for dtype in [torch.float32,torch.int32,torch.int64]:
 for shape in [(2,1344,64),(2,1344,5),(3,7,11),(1,1,1),(2,1,65),(1,17,1),(1,63,127),(2,65,3)]:
  a,b,c=shape
  parent=torch.arange((a+1)*(c+2)*(b+3),dtype=torch.int64).reshape(a+1,c+2,b+3)
  if dtype==torch.float32:parent=parent.to(dtype)*.125
  else:parent=parent.to(dtype)
  if dtype==torch.int64:parent=parent*4294967297
  gp=parent.to('ocl:0');source=gp[1:,1:1+c,2:2+b].permute(0,2,1);expected=parent[1:,1:1+c,2:2+b].permute(0,2,1)
  for rep in range(3):
   actual=source.clone(memory_format=torch.contiguous_format).cpu()
   torch.testing.assert_close(actual,expected,rtol=0,atol=0)
  dst=torch.full_like(parent,-123);gd=dst.to('ocl:0');target=gd[1:,1:1+c,2:2+b].permute(0,2,1)
  target.copy_(source);dst[1:,1:1+c,2:2+b].permute(0,2,1).copy_(expected)
  torch.testing.assert_close(gd.cpu(),dst,rtol=0,atol=0)
  count+=4
  print('PASS',str(dtype),shape,flush=True)
 x=torch.arange(21,dtype=dtype).reshape(1,7,3)
 g=x.to('ocl:0').expand(2,7,3)
 torch.testing.assert_close(g.clone(memory_format=torch.contiguous_format).cpu(),x.expand(2,7,3),rtol=0,atol=0);count+=1
print('PASS ALL',count,'copy comparisons',flush=True)
