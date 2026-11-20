"""Controlled full-HD wildlife training-step probe on labeled throughput fixtures.
No quality/epoch claim; original checkpoint is never overwritten.
"""
import argparse,copy,hashlib,json,os,time,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from math import gcd
import cv2,numpy as np,torch
import pytorch_ocl
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from fused_ema import OpenCLModelEMA
R=Path(os.environ.get('OPENCL_POSE_ROOT', '.')).resolve()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--batch',type=int,required=True);p.add_argument('--device',default='ocl:0')
p.add_argument('--steps',type=int,default=8);p.add_argument('--warmup',type=int,default=4)
p.add_argument('--check-loss',action='store_true');p.add_argument('--check-gradients',action='store_true');p.add_argument('--output',type=Path,required=True)
p.add_argument('--prefetch-input',action='store_true')
p.add_argument('--start-gate',type=Path)
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(7150);cv2.setNumThreads(1)
torch.backends.mkldnn.enabled=False;torch.backends.nnpack.set_flags(False)
device=torch.device(a.device);torch.ocl.set_device(device)
resolution=os.environ.get('BENCH_RESOLUTION','hd')
if resolution not in ('256','512','1024','hd'):
 raise ValueError(f'Unsupported benchmark resolution {resolution}')
display_h,width=(1080,1920) if resolution=='hd' else (int(resolution),)*2
height=(display_h+31)//32*32
manifest=json.loads((R/'fixtures/wildlife-256/manifest.json').read_text());frames=[];labels=[]
for item in manifest['images']:
 path=R/'fixtures/wildlife-256/images'/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 frames.append(cv2.resize(cv2.imread(str(path)),(width,display_h)))
 rows=np.loadtxt(R/'datasets/squirrel-wildlife-v2/valid/labels'/(path.stem+'.txt'),ndmin=2,dtype=np.float32)
 rows[:,[2,4]]*=display_h/height;labels.append(rows)
def pack_batch(step, *, padding_only=False):
 shape=(a.batch,3,height,width)
 if padding_only:
  packed=np.empty(shape,dtype=np.uint8)
  packed[:,:,display_h:,:]=114
 else:
  packed=np.full(shape,114,dtype=np.uint8)
 targets=[]
 for i in range(a.batch):
  index=(step*a.batch+i)%len(frames)
  packed[i,:,:display_h,:]=frames[index][:,:,::-1].transpose(2,0,1)
  targets.extend((i,*row) for row in labels[index].tolist())
 return packed,torch.tensor(targets,dtype=torch.float32).reshape(-1,6)

def upload_batch(batch):
 packed,target=batch
 return dict(img=torch.from_numpy(packed).to(device).float().div_(255),
             batch_idx=target[:,0],cls=target[:,1:2],bboxes=target[:,2:])

if a.prefetch_input:
 for step in range(len(frames)//gcd(len(frames),a.batch)):
  packed,target=pack_batch(step,padding_only=True)
  old_packed,old_target=pack_batch(step)
  assert np.array_equal(packed,old_packed)
  torch.testing.assert_close(target,old_target,rtol=0,atol=0)
 del packed,target,old_packed,old_target
 print('PASS exact CPU images and labels for every fixture batch',flush=True)
base=YOLO(str(R/'fixtures/wildlife-256/model.pt')).model.cpu().float().train();base.args=get_cfg()
for name,param in base.named_parameters():param.requires_grad_(not name.endswith('dfl.conv.weight'))
if hasattr(base,'criterion'):del base.criterion
cpu=copy.deepcopy(base) if a.check_gradients or a.check_loss else None
model=base.to(device)
from opencl_training_loss import install as install_opencl_loss
install_opencl_loss(model)
optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.9)
ema=OpenCLModelEMA(model);result=dict(batch=a.batch,device=a.device,input_shape=[a.batch,3,height,width],precision='FP32',
 source='20 labeled resized validation fixtures, no augmentation; controlled compute probe, not a dataset training epoch',
 checkpoint_sha256=hashlib.sha256((R/'fixtures/wildlife-256/model.pt').read_bytes()).hexdigest(),
 optimizer='SGD lr0.001 momentum0.9; clipping max_norm10; fused EMA',warmup=a.warmup,steps=[])
if a.check_gradients or a.check_loss:
 data=upload_batch(pack_batch(0,padding_only=a.prefetch_input));loss,items=model(data)
 if a.check_gradients:loss.backward()
 refdata={k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in data.items()}
 reference,refitems=cpu(refdata)
 if a.check_gradients:reference.backward()
 torch.testing.assert_close(loss.detach().cpu(),reference.detach(),rtol=1e-4,atol=1e-3)
 checks=[]
 for (name,param),(refname,ref) in zip(model.named_parameters(),cpu.named_parameters()):
  assert name==refname
  if param.grad is None or ref.grad is None:
   assert param.grad is None and ref.grad is None;continue
  actual=param.grad.detach().cpu();torch.testing.assert_close(actual,ref.grad,rtol=2e-2,atol=1e-3)
  checks.append(dict(parameter=name,max_absolute_error=float((actual-ref.grad).abs().max())))
 result['cpu_parity']=dict(loss=float(loss.detach().cpu()),cpu_loss=float(reference.detach()),gradient_parameters=len(checks),gradients=checks)
 optimizer.zero_grad(set_to_none=True);del cpu,data,refdata,loss,reference,items,refitems
 print('PASS CPU FULL-HD LOSS AND GRADIENTS',len(checks),flush=True)
total_steps=a.warmup+a.steps
pool=ThreadPoolExecutor(max_workers=1) if a.prefetch_input else None
pending=pool.submit(pack_batch,0,padding_only=True) if pool is not None else None
try:
 for step in range(total_steps):
  if step==a.warmup:
   (a.output.parent/'ready').touch()
   while a.start_gate and not a.start_gate.exists():time.sleep(.01)
   result['measured_start']=time.perf_counter()
  torch.ocl.synchronize(device);start=time.perf_counter()
  if pool is None:
   data=upload_batch(pack_batch(step))
  else:
   batch=pending.result()
   # The producer touches only host memory. OpenCL upload and all model
   # operations stay on this thread. At most one future batch is queued.
   pending=pool.submit(pack_batch,step+1,padding_only=True) if step+1<total_steps else None
   data=upload_batch(batch)
   del batch
  loss,items=model(data)
  value=float(loss.detach().cpu());components=items.detach().cpu()
  if not np.isfinite(value) or abs(value)>1e6 or not bool(torch.isfinite(components).all()) or bool((components < -1e-4).any()):raise RuntimeError('Nonfinite training loss')
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);gradient_norm=float(norm.detach().cpu())
  if not np.isfinite(gradient_norm):raise RuntimeError('Nonfinite training gradients')
  optimizer.step();optimizer.zero_grad(set_to_none=True);ema.update(model)
  torch.ocl.synchronize(device);elapsed=time.perf_counter()-start
  row=dict(step=step+1,seconds=elapsed,loss=value,loss_items=components.tolist(),gradient_norm=gradient_norm)
  result['steps'].append(row);print('STEP',json.dumps(row),flush=True)
  del data,loss,items,components,norm
finally:
 if pool is not None:pool.shutdown(wait=True)
result['input_prefetch']=a.prefetch_input
result['measured_end']=time.perf_counter()
samples=[row['seconds'] for row in result['steps'][a.warmup:]]
result.update(images_per_second=a.batch*len(samples)/sum(samples),seconds=sum(samples),measured_images=a.batch*len(samples),
 mean_step_seconds=statistics.mean(samples),allocator=torch.ocl.memory_stats(device),correctness_passed=True,
 note='Single GPU. Includes CPU packing/target preparation, GPU upload, train-mode forward, detection loss, backward, finite checks, gradient clipping, SGD and EMA. CPU gradient comparison outside timing; no video decode/augmentation/checkpoint/validation epoch.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps({k:result[k] for k in ('device','batch','images_per_second','seconds','measured_images','correctness_passed')}),flush=True)
