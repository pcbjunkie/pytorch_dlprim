import torch,pytorch_ocl
torch.manual_seed(6969);torch.set_num_threads(2)
for shape in [(),(0,3),(1,),(2,),(127,),(128,),(129,),(513,),(5,7,13)]:
 for dtype in [torch.float32,torch.int32,torch.int64,torch.bool]:
  x=(torch.rand(shape)>.6).to(dtype);g=x.to('ocl:0')
  torch.testing.assert_close(g.nonzero().cpu(),x.nonzero(),rtol=0,atol=0)
  print('PASS nonzero',shape,dtype,flush=True)
x=torch.arange(4*5*6*7.).reshape(4,5,6,7);g=x.to('ocl:0')
for mask in [torch.rand(4)>.4,torch.rand(4,5)>.5,torch.ones(4,5,6,dtype=torch.bool),torch.zeros(4,5,6,dtype=torch.bool)]:
 torch.testing.assert_close(g[mask.to('ocl:0')].cpu(),x[mask],rtol=0,atol=0)
 print('PASS mask',mask.shape,flush=True)
print('PASS legacy masked-selection regression',flush=True)
