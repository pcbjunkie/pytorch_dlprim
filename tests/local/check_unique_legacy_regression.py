"""Native small-list unique: compare all outputs and flags against ATen CPU."""
import itertools,torch,pytorch_ocl
torch.set_num_threads(2);torch.manual_seed(2059)
checks=0
for dtype in (torch.float32,torch.int32,torch.int64):
 samples=[torch.tensor(x,dtype=dtype) for x in ([],[0],[0,1],[0,0,1,1],[2,-1,2,0,-1],[5,5,5])]
 samples += [torch.tensor(7,dtype=dtype),torch.empty((0,3),dtype=dtype),torch.randint(-7,8,(7,9)).to(dtype).t(),torch.arange(40,dtype=dtype)[3:21],torch.randint(-16,17,(4096,)).to(dtype),torch.arange(4096,dtype=dtype)]
 if dtype==torch.float32:samples += [torch.tensor([float('nan'),1,float('nan'),float('inf'),-float('inf'),-0.,0.])]
 if dtype==torch.int32:samples += [torch.tensor([-2**31,2**31-1,-2**31,0],dtype=dtype)]
 if dtype==torch.int64:samples += [torch.tensor([-2**63,2**63-1,2**32+1,1,-2**32-1,2**63-1],dtype=dtype)]
 for x in samples:
  g=x.to('ocl:0')
  for sorted_,inverse,counts in itertools.product((False,True),repeat=3):
   ref=torch.ops.aten._unique2.default(x,sorted_,inverse,counts)
   actual=torch.ops.aten._unique2.default(g,sorted_,inverse,counts)
   torch.testing.assert_close(tuple(v.cpu() for v in actual),ref,rtol=0,atol=0,equal_nan=True)
   checks+=1
  print('PASS',dtype,tuple(x.shape),'stride',x.stride(),flush=True)
# Match YOLO's actual input: an image-ID column viewed on the GPU, rather
# than a noncontiguous CPU tensor materialized during upload.
base=torch.arange(36,dtype=torch.float32).reshape(6,6);base[:,0]=torch.tensor([0,1,0,1,1,0.])
gbase=base.to('ocl:0')
for start in (0,1,3):
 x=base[start:,0];g=gbase[start:,0]
 for sorted_,inverse,counts in itertools.product((False,True),repeat=3):
  ref=torch.ops.aten._unique2.default(x,sorted_,inverse,counts)
  actual=torch.ops.aten._unique2.default(g,sorted_,inverse,counts)
  torch.testing.assert_close(tuple(v.cpu() for v in actual),ref,rtol=0,atol=0,equal_nan=True)
  checks+=1
 print('PASS GPU-strided image-ID column',start,g.stride(),g.storage_offset(),flush=True)
print('PASS',checks,'CPU-referenced unique combinations',flush=True)
