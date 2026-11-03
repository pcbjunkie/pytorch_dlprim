"""Queued temporary lifetimes, aliasing, accounting and bounded retirement vs CPU."""
import gc,json,os
import torch,pytorch_ocl
assert os.environ.get('OPENCL_NO_MEM_CACHE')=='1'
assert os.environ.get('OPENCL_DEFERRED_RELEASE')=='1'
torch.set_num_threads(2);torch.manual_seed(2814)
torch.backends.mkldnn.enabled=False;torch.backends.nnpack.set_flags(False)
torch.ocl.empty_cache();torch.ocl.synchronize(0)
base=torch.ocl.memory_stats(0)

def run_chain():
 results=[];refs=[]
 for trial in range(12):
  cpu=torch.randn(8,16,32,32);gpu=cpu.to('ocl:0')
  # Reassignment repeatedly retires input/intermediate storage before later GPU
  # commands finish. Avoid host reads inside the chain.
  for i in range(19):
   cpu=cpu*.75+.125;gpu=gpu*.75+.125
  results.append(gpu);refs.append(cpu)
  del gpu
  # Force more allocations with different sizes while earlier results survive.
  junk=torch.empty(1025+trial*333,device='ocl:0');junk.fill_(trial);del junk
 for gpu,cpu in zip(results,refs):torch.testing.assert_close(gpu.cpu(),cpu,rtol=2e-6,atol=2e-6)
 print('PASS queued temporary chains and allocation churn',flush=True)
run_chain();gc.collect();torch.ocl.synchronize(0)

def pressure():
 cpu=torch.linspace(-1,1,9*1024*1024//4);gpu=cpu.to('ocl:0')
 for i in range(11):cpu=cpu*.5+.25;gpu=gpu*.5+.25
 torch.testing.assert_close(gpu.cpu(),cpu,rtol=0,atol=0)
 print('PASS large temporary chain and bounded retirement pressure',flush=True)
pressure();gc.collect();torch.ocl.synchronize(0)
assert torch.ocl.memory_allocated(0)==base['allocated_bytes.all.current']
assert torch.ocl.memory_reserved(0)==base['reserved_bytes.all.current']

def aliases():
 cpu=torch.arange(4096,dtype=torch.float32).reshape(64,64);gpu=cpu.to('ocl:0')
 view=gpu[1:63:2,2:62:3];ref=cpu[1:63:2,2:62:3].clone();del gpu
 outputs=[]
 for i in range(13):outputs.append((view+i).clone())
 del view
 for i,out in enumerate(outputs):torch.testing.assert_close(out.cpu(),ref+i,rtol=0,atol=0)
 print('PASS views retain storage, strided copies and queued consumers',flush=True)
aliases();gc.collect();torch.ocl.empty_cache();torch.ocl.synchronize(0)
stats=torch.ocl.memory_stats(0)
assert stats['allocated_bytes.all.current']==base['allocated_bytes.all.current'],stats
assert stats['reserved_bytes.all.current']==base['reserved_bytes.all.current'],stats
assert stats['deferred_bytes.all.current']==0,stats
assert stats['deferred_bytes.all.peak']>0,stats
assert stats['queue.marker_batches']>0 and stats['deferred.allocations']>0,stats
assert stats['queue.finish_calls']<stats['queue.sync_calls'],stats
print('ALLOCATOR_STATS',json.dumps(stats),flush=True)
print('PASS deferred buffer release regression',flush=True)
