"""Stable atomic-free indexed accumulation vs CPU, including views and long ties."""
import os,torch,pytorch_ocl
assert os.environ.get('DLPRIM_INDEX_SORT_REDUCE')=='1'
torch.set_num_threads(2);torch.manual_seed(2812)
for n in [0,1,63,64,65,127,128,129,513,1280,5200]:
 cpu=torch.randn(8193);idx=torch.randint(0,8193,(n,));v=torch.randn(n)
 ref=cpu.clone().index_put_((idx,),v,accumulate=True)
 gpu=cpu.to('ocl:0');gpu.index_put_((idx.to('ocl:0'),),v.to('ocl:0'),accumulate=True)
 torch.testing.assert_close(gpu.cpu(),ref,rtol=1e-5,atol=3e-6)
 print('PASS random repeated/empty/tail',n,flush=True)
# A segment longer than one workgroup/old update chunk must preserve addition order.
for n in [65,513,8193]:
 cpu=torch.zeros(8193);idx=torch.full((n,),137,dtype=torch.long)
 v=torch.ones(n)
 ref=cpu.clone().index_put_((idx,),v,accumulate=True)
 gpu=cpu.to('ocl:0');gpu.index_put_((idx.to('ocl:0'),),v.to('ocl:0'),accumulate=True)
 torch.testing.assert_close(gpu.cpu(),ref,rtol=0,atol=0)
 print('PASS long equal-key segment',n,flush=True)
for repeat in range(3):
 cpu=torch.randn(70,150);ref=cpu.clone();view=ref[1:65,2:130]
 ix=torch.randint(0,64,(128,));iy=torch.randint(-128,128,(128,));values=torch.randn(256)[::2]
 view.index_put_((ix,iy),values,accumulate=True)
 gpu=cpu.to('ocl:0');gview=gpu[1:65,2:130]
 gview.index_put_((ix.to('ocl:0'),iy.to('ocl:0')),values.to('ocl:0'),accumulate=True)
 torch.testing.assert_close(gpu.cpu(),ref,rtol=1e-5,atol=3e-6)
 print('PASS storage offset, physical strides, negative indices and neighbors',repeat,flush=True)
cpu=torch.randn(8,13,17);mask=torch.rand(8,13)>0.3
ref=cpu.clone();ref.index_put_((mask,),torch.tensor(2.),accumulate=True)
gpu=cpu.to('ocl:0');gpu.index_put_((mask.to('ocl:0'),),torch.tensor(2.,device='ocl:0'),accumulate=True)
torch.testing.assert_close(gpu.cpu(),ref,rtol=0,atol=0);print('PASS mask and scalar value broadcast',flush=True)
cpu=torch.zeros(16,1344,64);idx=torch.arange(28)*43;values=torch.randn(16,28,64)
ref=cpu.clone();ref.index_put_((torch.arange(16)[:,None],idx[None,:]),values,accumulate=True)
gpu=cpu.to('ocl:0');gpu.index_put_((torch.arange(16,device='ocl:0')[:,None],idx.to('ocl:0')[None,:]),values.to('ocl:0'),accumulate=True)
torch.testing.assert_close(gpu.cpu(),ref,rtol=0,atol=0);print('PASS YOLO-shaped sparse gradient',flush=True)
print('PASS sorted index accumulation regression',flush=True)
