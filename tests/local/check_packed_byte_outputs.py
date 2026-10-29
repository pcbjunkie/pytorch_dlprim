"""Byte-output CPU checks: masks, boundaries, strides, offsets and neighboring bytes."""
import os,operator,torch,pytorch_ocl
assert os.environ.get('DLPRIM_PACKED_BYTE_OUTPUTS')=='1'
torch.set_num_threads(2);torch.manual_seed(64128)
ops=[operator.lt,operator.le,operator.gt,operator.ge,operator.eq,operator.ne]
checks=0
for n in [0,1,3,4,5,63,64,65,127,128,129,255,256,257,13440]:
 for dtype in [torch.float32,torch.int32,torch.uint8,torch.int8]:
  x=(torch.arange(n)%5).to(dtype);g=x.to('ocl:0')
  for op in ops:
   for other,gother in [(1,1),(torch.ones_like(x),torch.ones_like(x).to('ocl:0'))]:
    torch.testing.assert_close(op(g,gother).cpu(),op(x,other),rtol=0,atol=0);checks+=1
 print('PASS comparison length',n,flush=True)
for value in [0.,.5,1.,2.]:
 g=torch.tensor(value).to('ocl:0');actual=max(g,1);actual=float(actual.cpu()) if isinstance(actual,torch.Tensor) else actual
 assert actual==max(value,1);checks+=1
print('PASS Python max scalar loss guard',flush=True)
x=torch.tensor([float('nan'),float('inf'),-float('inf'),-0.,0.,1.]);g=x.to('ocl:0')
for op in ops:torch.testing.assert_close(op(g,1).cpu(),op(x,1),rtol=0,atol=0);checks+=1
for n in [1,3,4,5,17,65]:
 cpu=torch.ones(2*n+5,dtype=torch.bool);gpu=cpu.to('ocl:0')
 x=torch.arange(n,dtype=torch.float32)%3;gx=x.to('ocl:0')
 torch.lt(gx,1,out=gpu[2:2+2*n:2]);torch.lt(x,1,out=cpu[2:2+2*n:2])
 torch.testing.assert_close(gpu.cpu(),cpu,rtol=0,atol=0);checks+=1
 for offset in range(4):
  a=torch.ones(n+7,dtype=torch.bool);b=a.to('ocl:0')
  torch.gt(gx,1,out=b[offset:offset+n]);torch.gt(x,1,out=a[offset:offset+n])
  torch.testing.assert_close(b.cpu(),a,rtol=0,atol=0);checks+=1
print('PASS out offsets and strided writes preserve neighbors',flush=True)
for dtype,values in [(torch.uint8,[0,1,127,128,254,255]),(torch.int8,[-128,-1,0,1,126,127])]:
 x=torch.tensor(values,dtype=dtype);g=x.to('ocl:0')
 for actual,expected in [(g+2,x+2),(~g,~x),(g.clone(),x.clone())]:torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0);checks+=1
 x=x.repeat(4,1);g=x.to('ocl:0');torch.testing.assert_close(g.t().contiguous().cpu(),x.t().contiguous(),rtol=0,atol=0);checks+=1
m=torch.tensor([0,1,1,0,1],dtype=torch.bool);g=m.to('ocl:0')
for actual,expected in [(~g,~m),(g&~g,m&~m),(g|~g,m|~m),(g.clone(),m.clone())]:torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0);checks+=1
print('PASS',checks,'CPU comparisons/byte operations',flush=True)
