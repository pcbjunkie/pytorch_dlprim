"""Regression for the sparse YOLO mask whose fourth coordinate was corrupted."""
import torch,pytorch_ocl
torch.set_num_threads(2)
positions=[459,460,491,1125,1126,1141,1142,1143,1157,1158,2644,2645,2651,2652,2653,2654,2659,2661,2662,2669]
mask=torch.zeros(2688,dtype=torch.bool);mask[positions]=True;mask=mask.reshape(2,1344)
x=(torch.arange(2*1344*64,dtype=torch.float32)+13).reshape(2,1344,64)
coords=mask.nonzero();expected=x[mask]
for i in range(200):
 g=x.to('ocl:0');m=mask.to('ocl:0')
 torch.testing.assert_close(m.nonzero().cpu(),coords,rtol=0,atol=0)
 torch.testing.assert_close(g[m].cpu(),expected,rtol=0,atol=0)
 if i%25==0:print('PASS',i,flush=True)
print('PASS200 sparse mask coordinates and gathers',flush=True)
