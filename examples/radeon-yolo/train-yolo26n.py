"""Controlled full-HD wildlife training-step probe on labeled throughput fixtures.
No quality/epoch claim; original checkpoint is never overwritten.
"""
import argparse,copy,hashlib,json,os,time,statistics,sys
from pathlib import Path
import cv2,numpy as np,torch
import pytorch_ocl
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(os.environ.get('OPENCL_POSE_ROOT', '.'))/'runs/ultralytics-config'))
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
R=Path(os.environ.get('OPENCL_POSE_ROOT', '.')).resolve()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--family',choices=['yolo26','yolox'],required=True)
p.add_argument("--parity-fixture",type=int,default=0)
p.add_argument('--batch',type=int,required=True);p.add_argument('--device',default='ocl:0')
p.add_argument('--steps',type=int,default=8);p.add_argument('--warmup',type=int,default=4)
p.add_argument('--check-loss',action='store_true');p.add_argument('--check-gradients',action='store_true');p.add_argument('--output',type=Path,required=True)
p.add_argument("--capture-profile",action="store_true");p.add_argument("--ready",type=Path);p.add_argument("--start-file",type=Path);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(7150);cv2.setNumThreads(1)
torch.backends.mkldnn.enabled=False;torch.backends.nnpack.set_flags(False)
device=torch.device(a.device);torch.ocl.set_device(device);torch.ocl.enable_profiling(device)
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
def pack_batch(step, padding_only=False):
 packed=np.empty((a.batch,3,height,width),dtype=np.uint8)
 (packed[:,:,display_h:,:] if padding_only else packed).fill(114)
 targets=[]
 for i in range(a.batch):
  index=(step*a.batch+i)%len(frames);packed[i,:,:display_h,:]=(frames[index] if a.family=='yolox' else frames[index][:,:,::-1]).transpose(2,0,1)
  rows=labels[index];targets.extend([[i,*row.tolist()] for row in rows])
 target=torch.tensor(targets,dtype=torch.float32).reshape(-1,6)
 if a.family=='yolox':
  counts=[int((target[:,0]==i).sum()) for i in range(a.batch)];dense=torch.zeros(a.batch,max(1,max(counts)),5)
  for i,n in enumerate(counts):
   if n:
    row=target[target[:,0]==i,1:].clone();row[:,[1,3]]*=width;row[:,[2,4]]*=height;dense[i,:n]=row
  return packed,dense
 return packed,target

def upload_batch(host_batch):
 packed,target=host_batch
 image=torch.from_numpy(packed).to(device).float()
 if a.family=='yolox':
  return dict(img=image,targets=target.to(device))
 return dict(img=image.div_(255),batch_idx=target[:,0],cls=target[:,1:2],bboxes=target[:,2:])

def batch_at(step):
 return upload_batch(pack_batch(step))

if a.family=='yolo26':
 from ultralytics.nn.tasks import DetectionModel
 checkpoint=R/'models/hd-comparison/yolo26n.pt'
 pretrained=YOLO(str(checkpoint)).model.cpu().float()
 base=DetectionModel(copy.deepcopy(pretrained.yaml),nc=5,verbose=False).cpu().float().train()
 base.load(pretrained,verbose=False);base.args=get_cfg();del pretrained
else:
 sys.path.insert(0,str(R/'sources/YOLOX'))
 from yolox.exp import get_exp
 checkpoint=R/'models/hd-comparison/yolox_nano.pth'
 exp=get_exp(None,'yolox-nano');exp.num_classes=5;base=exp.get_model().cpu().float().train()
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'];current=base.state_dict()
 compatible={k:v for k,v in state.items() if k in current and current[k].shape==v.shape}
 base.load_state_dict(compatible,strict=False)
 del state,current,compatible
base.names={0:'squirrel',1:'bobcat',2:'deer',3:'opossum',4:'raccoon'}
def train_forward(network,data):
 if a.family=='yolox':
  output=network(data['img'],data['targets']);loss=output['total_loss']
  items=torch.stack([torch.as_tensor(output[k],device=data['img'].device) for k in ['iou_loss','conf_loss','cls_loss','l1_loss']])
 else:
  loss,items=network(data)
  if isinstance(items,dict):items=torch.stack(list(items.values()))
 return loss.sum(),items
for name,param in base.named_parameters():param.requires_grad_(not name.endswith('dfl.conv.weight'))
if hasattr(base,'criterion'):del base.criterion
cpu=copy.deepcopy(base) if a.check_gradients or a.check_loss else None
model=base.to(device)
from opencl_training_loss import install as install_opencl_loss
if a.family=='yolo26':
 install_opencl_loss(model)
 if not hasattr(model,"criterion"):model.criterion=model.init_criterion()
 # The OpenCL dim-max supports int32 but not the assigner's int8 0/1 mask.
 from types import MethodType
 def select_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
  multiple = (mask_pos.sum(-2).unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
  best = overlaps.max(1).indices
  selected = torch.zeros_like(mask_pos).scatter_(1, best.unsqueeze(1), 1)
  mask_pos = torch.where(multiple, selected, mask_pos)
  if self.topk2 != self.topk:
   ids = torch.topk(align_metric * mask_pos, self.topk2, dim=-1).indices
   mask_pos *= torch.zeros_like(mask_pos).scatter_(-1, ids, 1)
  foreground, indices = mask_pos.to(torch.int32).max(-2)
  return indices, foreground.to(mask_pos.dtype), mask_pos
 for branch in (model.criterion.one2many, model.criterion.one2one):
  branch.assigner.select_highest_overlaps=MethodType(select_overlaps,branch.assigner)

from metrics_trial import install as install_metrics_trial
install_metrics_trial(model.criterion)
from preprocess_trial import install as install_preprocess_trial
stage_targets = install_preprocess_trial(model)
from scatter_trial import install as install_scatter_trial
install_scatter_trial(model)
from target_trial import install as install_target_trial
install_target_trial(model)
from bn_silu_trial import install as install_bn_silu
print("BN SILU BLOCKS",install_bn_silu(model),flush=True)

if os.environ.get('PROFILE_V26_CIOU_L1') == '1':
 from ciou_l1_trial import install as install_ciou_l1
 install_ciou_l1(model)

if os.environ.get('PROFILE_V26_TAL_MASK') == '1':
 from tal_mask_trial import install as install_tal_mask
 install_tal_mask(model)

optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.9)
result=dict(batch=a.batch,device=a.device,input_shape=[a.batch,3,height,width],precision='FP32',
 source='20 labeled resized validation fixtures, no augmentation; controlled compute probe, not a dataset training epoch',
 family=a.family,classes=5,initialization='COCO pretrained compatible weights; five-class classifier reinitialized, not a wildlife accuracy checkpoint',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
 optimizer='SGD lr0.001 momentum0.9; clipping max_norm10',warmup=a.warmup,steps=[])
if a.check_gradients or a.check_loss:
 data=batch_at(a.parity_fixture);loss,items=train_forward(model,data)
 if a.check_gradients:loss.backward()
 refdata={k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in data.items()}
 reference,refitems=train_forward(cpu,refdata)
 if a.check_gradients:reference.backward()
 torch.testing.assert_close(loss.detach().cpu(),reference.detach(),rtol=1e-4,atol=1e-3)
 checks=[]
 for (name,param),(refname,ref) in zip(model.named_parameters(),cpu.named_parameters()):
  assert name==refname
  if param.grad is None or ref.grad is None:
   assert param.grad is None and ref.grad is None;continue
  actual=param.grad.detach().cpu();torch.testing.assert_close(actual,ref.grad,rtol=2e-2,atol=1e-3,msg=lambda msg: f"{name}: {msg}")
  checks.append(dict(parameter=name,max_absolute_error=float((actual-ref.grad).abs().max())))
 result['cpu_parity']=dict(loss=float(loss.detach().cpu()),cpu_loss=float(reference.detach()),gradient_parameters=len(checks),gradients=checks)
 optimizer.zero_grad(set_to_none=True);del cpu,data,refdata,loss,reference,items,refitems
 print('PASS CPU FULL-HD LOSS, gradient tensors checked:',len(checks),flush=True)

from contextlib import nullcontext
from collections import defaultdict
import csv
def sync():torch.ocl.synchronize(device)
def traced_loss(data, preds):
 # Capture Python-level transfers during one warmup. Timed updates never use this wrapper.
 rows = []
 original_to = torch.Tensor.to
 def capture(tensor, *args, **kwargs):
  start = time.perf_counter_ns()
  output = original_to(tensor, *args, **kwargs)
  elapsed = time.perf_counter_ns() - start
  if output is not tensor:
   caller = sys._getframe(1)
   rows.append({
       'file': Path(caller.f_code.co_filename).name, 'function': caller.f_code.co_name,
       'line': caller.f_lineno, 'source': str(tensor.device), 'target': str(output.device),
       'source_dtype': str(tensor.dtype), 'target_dtype': str(output.dtype),
       'elements': tensor.numel(), 'host_ms': elapsed / 1e6,
   })
  return output
 torch.Tensor.to = capture
 try:
  return model.loss(data, preds=preds)
 finally:
  torch.Tensor.to = original_to
  (a.output.parent / 'loss-python-transfers.json').write_text(json.dumps(rows, indent=2))

def whole(step, mark, host_batch=None):
 with mark('input'):
  data=batch_at(step) if host_batch is None else upload_batch(host_batch)
  stage_targets(data)
 with mark('forward'):
  preds=model(data['img'])
 with mark('loss'):
  transfer_trace = a.output.parent / 'loss-python-transfers.json'
  if a.capture_profile and step == 0 and not transfer_trace.exists():
   loss,items=traced_loss(data,preds)
  else:
   loss,items=model.loss(data,preds=preds)
  loss=loss.sum()
 with mark('loss_readback'):
  value=float(loss.detach().cpu())
  assert np.isfinite(value)
 with mark('backward'):
  loss.backward()
 with mark('clip_and_step'):
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
  assert np.isfinite(float(norm.detach().cpu()))
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)
 sync()
 return value
plain=[]
window_start=None
prefetch = os.environ.get('PROFILE_V26_PREFETCH') == '1'
if prefetch:
 from math import gcd
 from concurrent.futures import ThreadPoolExecutor
 for step in range(len(frames) // gcd(len(frames), a.batch)):
  old_images, old_targets = pack_batch(step)
  images, targets = pack_batch(step, padding_only=True)
  assert np.array_equal(images, old_images) and torch.equal(targets, old_targets)
 del old_images,old_targets,images,targets
 print('PASS HOST BATCH PARITY',flush=True)
pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
total_steps = a.warmup+a.steps
try:
 pending = pool.submit(pack_batch,0,True) if pool is not None else None
 for step in range(total_steps):
  sync()
  if step==a.warmup:
   if a.ready:
    a.ready.touch()
    deadline=time.monotonic()+180
    while not a.start_file.exists():
     if time.monotonic()>deadline: raise RuntimeError("paired start timeout")
     time.sleep(.005)
   queue_start=torch.ocl.memory_stats(device)
   window_start=time.monotonic()
  begin=time.perf_counter()
  host_batch=pending.result() if pending is not None else None
  if pool is not None:
   pending=(pool.submit(pack_batch,step+1,True) if step+1<total_steps else None)
  whole(step,lambda _:nullcontext(),host_batch)
  del host_batch
  if step>=a.warmup:plain.append((time.perf_counter()-begin)*1000/a.batch)
 window_end=time.monotonic()
finally:
 if pool is not None:
  pool.shutdown(cancel_futures=True)
queue_end=torch.ocl.memory_stats(device)
timed_queue={k:(queue_end[k]-queue_start[k])/len(plain) for k in queue_start if k.startswith('queue.') }
result={'queue_per_update':timed_queue,'window_start':window_start,'window_end':window_end,'steps':len(plain),'batch':a.batch,'input_shape':[a.batch,3,height,width],
        'cpu_parity':result.get('cpu_parity'),'input_prefetch':prefetch,'unprofiled_ms_per_image':plain,
        'images_per_second':1000/statistics.mean(plain),'whole':[],'staged':[]}
if not a.capture_profile:
 a.output.write_text(json.dumps(result,indent=2))
 print('TRAIN',a.batch,result['images_per_second'],flush=True)
 raise SystemExit(0)
for i in range(3):
 sync();start=time.perf_counter()
 before=torch.ocl.memory_stats(device)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
  with torch.ocl.profile(device,str(a.output.parent/f'whole-{i}.csv')):
   begin=time.perf_counter()
   whole(a.warmup+i,lambda name:torch.profiler.record_function('stage_'+name))
   wall=(time.perf_counter()-begin)*1000
 after=torch.ocl.memory_stats(device)
 scopes=defaultdict(lambda:[0,0.,0.])
 waits=defaultdict(lambda:[0,0.])
 for event in prof.events():
  parent=event.cpu_parent
  while parent is not None and not parent.name.startswith('stage_'):parent=parent.cpu_parent
  if parent is None:continue
  if parent.name=='stage_loss' and event.name in ('aten::_local_scalar_dense','aten::_copy_from'):
   ancestor=event.cpu_parent
   route=[]
   while ancestor is not None and ancestor!=parent:
    route.append(ancestor.name)
    ancestor=ancestor.cpu_parent
   entry=waits[(event.name,' > '.join(reversed(route)))]
   entry[0]+=1;entry[1]+=event.self_cpu_time_total/1000
  entry=scopes[(parent.name,event.name)]
  entry[0]+=1;entry[1]+=event.self_cpu_time_total/1000;entry[2]+=event.cpu_time_total/1000
 rows=[dict(stage=k[0],op=k[1],calls=v[0],self_ms=v[1],inclusive_ms=v[2]) for k,v in scopes.items()]
 (a.output.parent/f'cpu-{i}.json').write_text(json.dumps(rows,indent=2))
 (a.output.parent/f'waits-{i}.json').write_text(json.dumps([dict(op=k[0],parent=k[1],calls=v[0],self_ms=v[1]) for k,v in waits.items()],indent=2))
 result['whole'].append(dict(wall_ms=wall,queue={k:after[k]-before[k] for k in before if k.startswith('queue.')}))
for i in range(5):
 stages={}
 from contextlib import contextmanager
 @contextmanager
 def measure(name):
  sync();before=torch.ocl.memory_stats(device);start=time.perf_counter()
  with torch.ocl.profile(device,str(a.output.parent/f'stage-{i}-{name}.csv')):
   yield
   sync()
  after=torch.ocl.memory_stats(device)
  stages[name]=dict(wall_ms=(time.perf_counter()-start)*1000,
                    queue={k:after[k]-before[k] for k in before if k.startswith('queue.')})
 whole(a.warmup+i,measure)
 result['staged'].append(stages)
a.output.write_text(json.dumps(result,indent=2))
print('WALL',statistics.median(plain),flush=True)
