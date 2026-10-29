"""Validate no-cache allocation granularity, accounting, aliases and GPU results."""
import gc,os,torch,pytorch_ocl
assert os.environ.get('OPENCL_TIGHT_ALLOC')=='1'
assert os.environ.get('OPENCL_NO_MEM_CACHE')=='1'
torch.set_num_threads(2)
warm=torch.empty(1,device='ocl:0');del warm;gc.collect();torch.ocl.synchronize(0)
base=torch.ocl.memory_reserved(0)
for n in [1,1023,1024,1025,2049,9*1024*1024//4,12*1024*1024//4,17*1024*1024//4]:
 cpu=(torch.arange(n,dtype=torch.float32)%31)-15
 x=cpu.to('ocl:0')
 byte_count=n*4;expected=(byte_count+4095)//4096*4096 if byte_count>=4096 else 1<<(byte_count-1).bit_length()
 actual=torch.ocl.memory_reserved(0)-base
 assert actual==expected,(n,byte_count,actual,expected)
 alias=x[1::2]
 assert torch.ocl.memory_reserved(0)-base==expected
 y=x*2+1
 torch.testing.assert_close(y.cpu(),cpu*2+1,rtol=0,atol=0)
 # Exercise a strided input without allowing its backing allocation to disappear.
 if alias.numel():
  z=alias+3
  torch.testing.assert_close(z.cpu(),cpu[1::2]+3,rtol=0,atol=0)
  del z
 del x,y,alias
 gc.collect();torch.ocl.synchronize(0)
 assert torch.ocl.memory_reserved(0)==base,torch.ocl.memory_stats(0)
 print('PASS requested',byte_count,'reserved',actual,'saved_vs_power2',(1<<(byte_count-1).bit_length())-actual,flush=True)
print('PASS tight allocation regression',flush=True)
