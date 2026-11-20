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
if os.environ.get('PROFILE_BN_SILU') == '1':
 from types import MethodType
 from bn_silu_trial import TrainingBNSiLU
 from yolox.models.network_blocks import BaseConv
 def fused_forward(block, x):
  bn = block.bn
  if (os.environ.get('PROFILE_BN_SILU') != '1' or not block.training or not bn.training
      or x.device.type != 'ocl' or x.dtype != torch.float32):
   return BaseConv.forward(block, x)
  value = block.conv(x)
  bn.num_batches_tracked.add_(1)
  return TrainingBNSiLU.apply(value, bn.weight, bn.bias, bn.running_mean,
                              bn.running_var, bn.momentum, bn.eps)
 count = 0
 for block in model.modules():
  if (type(block) is BaseConv and type(block.bn) is torch.nn.BatchNorm2d
      and type(block.act) is torch.nn.SiLU and block.bn.affine
      and block.bn.track_running_stats and block.bn.momentum is not None):
   block.forward = MethodType(fused_forward, block)
   count += 1
 print('BN SILU BLOCKS', count, flush=True)
if os.environ.get('PROFILE_X_GEOMETRY') == '1':
 from geometry_trial import install
 install(model.head)
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

if os.environ.get('PROFILE_X_GRAD_REPLAY') == '1':
 trial_flags = dict(PROFILE_BN_SILU='1', PROFILE_X_GEOMETRY='1',
                    DLPRIM_DW_STRIDE2_INPUT_GATHER='1',
                    DLPRIM_WINOGRAD_FILTER_COMPACT='1', DLPRIM_WINOGRAD_FILTER_PARTS='64')
 saved_buffers = {name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()}
 rows = []
 for fixture in (0, 1):
  replays = []
  for enabled in (False, True):
   os.environ.update({key: val if enabled else '0' for key, val in trial_flags.items()})
   with torch.no_grad():
    for name, buffer in model.named_buffers():
     buffer.copy_(saved_buffers[name])
   model.zero_grad(set_to_none=True)
   loss, items = train_forward(model, batch_at(fixture))
   loss.backward()
   replays.append((loss.detach().cpu(), {
       name: param.grad.detach().cpu() for name, param in model.named_parameters()
       if param.grad is not None}))
   del loss, items
  reference, candidate = replays
  torch.testing.assert_close(candidate[0], reference[0], rtol=1e-4, atol=1e-3)
  assert candidate[1].keys() == reference[1].keys()
  failures = []
  for name, grad in candidate[1].items():
   if not torch.allclose(grad, reference[1][name], rtol=2e-2, atol=1e-3):
    failures.append(name)
  rows.append(dict(fixture=fixture, parameters=len(candidate[1]), failures=failures,
                   loss_delta=float(candidate[0] - reference[0]),
                   max_abs=max(float((grad - reference[1][name]).abs().max())
                               for name, grad in candidate[1].items())))
  print(rows[-1], flush=True)
 a.output.write_text(json.dumps(rows, indent=2))
 assert not any(row['failures'] for row in rows)
 raise SystemExit(0)

from contextlib import ExitStack, nullcontext
from collections import defaultdict
import csv
def sync():torch.ocl.synchronize(device)
def whole(step, mark, host_batch=None):
 with mark('input'):
  data=batch_at(step) if host_batch is None else upload_batch(host_batch)
 with mark('forward'):
  features=model.backbone(data['img'])
 original_get_losses = model.head.get_losses
 with ExitStack() as phase:
  phase.enter_context(mark('head'))
  def get_losses(*args, **kwargs):
   phase.close()
   phase.enter_context(mark('loss'))
   return original_get_losses(*args, **kwargs)
  model.head.get_losses = get_losses
  try:
   loss, *_ = model.head(features, data['targets'], data['img'])
  finally:
   del model.head.get_losses
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
prefetch = os.environ.get('PROFILE_X_PREFETCH') == '1'
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
