import torch,pytorch_ocl
torch.set_num_threads(2)
for n in [1,5,65,1344]:
 for count in [1,10,32]:
  x=(torch.arange(2*n).reshape(2,1,n)%7-3).to(torch.int8)
  idx=(torch.arange(2*count).reshape(2,1,count)%n).long()
  src=torch.ones_like(idx,dtype=torch.int8)
  gx=x.to('ocl:0');gi=idx.to('ocl:0');gs=src.to('ocl:0')
  gx.scatter_add_(-1,gi,gs)
  torch.testing.assert_close(gx.cpu(),x.scatter_add(-1,idx,src),rtol=0,atol=0)
 print('PASS scatter',n,flush=True)
print('PASS int8 scatter repeated indices',flush=True)
