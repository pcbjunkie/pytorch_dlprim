"""FP32 full-HD detector-input probes; fixed fixture-derived decoded HD frames.

These are throughput/correctness fixtures, not native-HD accuracy qualification
or a camera/video decoder benchmark. Padding takes 1920x1080 to1920x1088.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import csv
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
import pytorch_ocl

ROOT=Path(os.environ.get('OPENCL_POSE_ROOT', '.')).resolve()
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/hd-throughput-20261115/ultralytics-config'))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--family',choices=['wildlife','yolo26','yolox'],required=True)
 p.add_argument('--batch',type=int,default=1)
 p.add_argument('--resolution',choices=['256','512','1024','hd'],default='hd')
 p.add_argument('--device',default='ocl:0')
 p.add_argument('--pci-device',help='PCI BDF for memory counters when using a different GPU')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--iterations',type=int,default=10)
 p.add_argument('--warmup',type=int,default=2)
 p.add_argument('--check-images',type=int,default=2)
 p.add_argument('--seconds',type=float,default=0)
 p.add_argument('--forward-seconds',type=float)
 p.add_argument('--ready',type=Path)
 p.add_argument('--start-file',type=Path)
 p.add_argument('--raw-only',action='store_true')
 p.add_argument('--postprocess-device',choices=['gpu','cpu'],default='gpu')
 p.add_argument('--cpu-head-layout',choices=['anchors-first','channels-first'],default='anchors-first')
 p.add_argument('--copy-optimizations',action='store_true')
 p.add_argument('--fuse-input-focus',action='store_true')
 p.add_argument('--depthwise-vector4',action='store_true')
 p.add_argument('--depthwise-stride2',action='store_true')
 p.add_argument('--depthwise-layout',choices=['baseline','columns','32x4','64x1','16x4','32x8'],default='baseline')
 p.add_argument('--small-k',action='store_true')
 p.add_argument('--pointwise-reduce',action='store_true')
 p.add_argument('--gemm-experiments',default='')
 p.add_argument('--gemm-n',choices=['0','64','128','256','balanced'],default='0')
 p.add_argument('--fuse-sppf',action='store_true')
 p.add_argument('--fuse-conv-silu',action='store_true')
 p.add_argument('--compact-winograd',action='store_true')
 p.add_argument('--winograd-mode',choices=['baseline','compact','banked','small'],default='baseline')
 p.add_argument('--gemm-pad-lds',action='store_true')
 p.add_argument('--input-packing',choices=['per-image','batch'],default='per-image')
 p.add_argument('--prefetch-input',action='store_true')
 p.add_argument('--profile-stages',type=int,default=0)
 p.add_argument('--profile-only',action='store_true')
 p.add_argument('--gpu-detections',action='store_true')
 p.add_argument('--reuse-readback',action='store_true')
 p.add_argument('--readback-chunk-mib',type=int,choices=[0,1,4,8],default=0)
 a=p.parse_args()
 from gemm_experiment_flags import apply as apply_gemm_experiments
 apply_gemm_experiments(a.gemm_experiments)
 if a.fuse_input_focus and (a.family!='yolox' or a.input_packing!='batch'):p.error('Fused input Focus requires YOLOX and batch packing')
 if a.profile_only and a.profile_stages<=0:p.error('profile-only requires positive profile-stages')
 if a.gpu_detections and a.family not in ('wildlife','yolox') and (a.family!='yolo26' or a.postprocess_device!='cpu' or a.cpu_head_layout!='channels-first'):
  p.error('GPU detections requires YOLO26 dense channels-first head')
 if a.readback_chunk_mib:a.reuse_readback=True
 if a.prefetch_input and a.input_packing!='batch':p.error('prefetch requires batch input packing')
 if a.copy_optimizations:
  os.environ['DLPRIM_COALESCE_COPY']='1';os.environ['DLPRIM_DIRECT_CAT']='1'
 if a.depthwise_vector4:
  os.environ['DLPRIM_DEPTHWISE_VECTOR4']='1';a.depthwise_stride2=True
 if a.depthwise_stride2:os.environ['DLPRIM_DEPTHWISE_STRIDE2']='1'
 os.environ.pop('DLPRIM_DEPTHWISE_LAYOUT',None)
 if a.depthwise_layout!='baseline':os.environ['DLPRIM_DEPTHWISE_LAYOUT']=a.depthwise_layout
 if a.small_k:os.environ['DLPRIM_SMALL_K']='1'
 if a.pointwise_reduce:os.environ['DLPRIM_POINTWISE_REDUCE']='1'
 if a.gemm_n!='0':os.environ['DLPRIM_FORWARD_GEMM_N']=a.gemm_n
 if a.compact_winograd:os.environ['DLPRIM_WINOGRAD_COMPACT']='1'
 if a.winograd_mode!='baseline':os.environ['DLPRIM_WINOGRAD_COMPACT']={'compact':'1','banked':'2','small':'small'}[a.winograd_mode]
 if a.gemm_pad_lds:os.environ['DLPRIM_GEMM_PAD_LDS']='1'
 torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;torch.backends.nnpack.set_flags(False);cv2.setNumThreads(1)
 torch.manual_seed(20261115)
 if a.family=='yolox':
  sys.path.insert(0,str(ROOT/'sources/YOLOX'))
  from yolox.exp import get_exp
  from yolox.utils import fuse_model,postprocess
  checkpoint=ROOT/'models/hd-comparison/yolox_nano.pth'
  reference=get_exp(None,'yolox-nano').get_model()
  reference.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
  with torch.no_grad():reference=fuse_model(reference.eval()).float()
  def nms(raw):return postprocess(raw.clone(),80,.25,.7,class_agnostic=False)
 else:
  from ultralytics import YOLO
  if a.family=='wildlife':
   from ultralytics.utils.ops import non_max_suppression
  checkpoint=ROOT/('fixtures/wildlife-256/model.pt' if a.family=='wildlife' else 'models/hd-comparison/yolo26n.pt')
  reference=YOLO(str(checkpoint)).model.cpu().float().eval()
  if a.family=='yolo26':reference.model[-1].end2end=True
  reference.fuse(verbose=False)
  if a.family=='yolo26':
   assert reference.model[-1].end2end,'NMS-free head was not selected'
   cpu_postprocess=reference.model[-1].postprocess
   if a.postprocess_device=='cpu':
    # Intentional model/pipeline boundary; no implicit backend fallback.
    reference.model[-1].postprocess=(lambda predictions:predictions.transpose(1,2)) if a.cpu_head_layout=='channels-first' else (lambda predictions:predictions)
    def nms(raw):
     value=raw.transpose(1,2) if a.cpu_head_layout=='channels-first' else raw
     return [r[r[:,4]>=.25] for r in cpu_postprocess(value)]
   else:
    def nms(raw):return [r[r[:,4]>=.25] for r in raw]
  else:
   def nms(raw):return non_max_suppression(raw.clone(),conf_thres=.25,iou_thres=.7,nc=5,max_det=300)
 device=torch.device(a.device)
 torch.ocl.set_device(device)
 if a.profile_stages:torch.ocl.enable_profiling(device)
 network=copy.deepcopy(reference).to(device)
 fused_blocks=[]
 if a.fuse_conv_silu:
  from hd_inference_fusions import fuse_conv_silu
  fused_blocks=fuse_conv_silu(network)
 sppf_blocks=[]
 if a.fuse_sppf:
  from hd_inference_fusions import fuse_sppf
  sppf_blocks=fuse_sppf(network)
 focus_blocks=[]
 if a.fuse_input_focus:
  from hd_inference_fusions import accept_precomputed_focus
  focus_blocks=accept_precomputed_focus(network)
 display_h, width = (1080,1920) if a.resolution=='hd' else (int(a.resolution),)*2
 height = (display_h+31)//32*32
 manifest_path=ROOT/'fixtures/wildlife-256/manifest.json'
 manifest=json.loads(manifest_path.read_text());frames=[];inputs=[]
 for item in manifest['images'][:max(a.check_images,2)]:
  path=ROOT/'fixtures/wildlife-256/images'/item['path'];assert digest(path)==item['sha256']
  image=cv2.imread(str(path));assert image is not None
  # Derived HD source frames preserve fixture identity in metadata; no accuracy claim.
  frames.append(cv2.resize(image,(width,display_h),interpolation=cv2.INTER_LINEAR))
 def prepare(frame,target):
  padded=np.full((height,width,3),114,dtype=np.uint8);padded[:display_h]=frame
  image=padded if a.family=='yolox' else padded[:,:,::-1]
  tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).to(target).float()
  return tensor if a.family=='yolox' else tensor.div_(255)
 def predict(model,tensor):
  value=model(tensor)
  return value[0] if isinstance(value,(list,tuple)) else value
 def pack_batch(selected):
  if a.fuse_input_focus:return np.stack(selected)
  packed=np.full((len(selected),3,height,width),114,dtype=np.uint8)
  for i,frame in enumerate(selected):
   packed[i,:,:display_h,:]=(frame if a.family=='yolox' else frame[:,:,::-1]).transpose(2,0,1)
  return packed
 def upload_batch(packed):
  if a.fuse_input_focus:return torch.ops.pytorch_ocl.focus_bgr_u8(torch.from_numpy(packed).to(device))
  value=torch.from_numpy(packed).to(device).float()
  return value if a.family=='yolox' else value.div_(255)
 def prepare_batch(selected):return upload_batch(pack_batch(selected))
 def sync():torch.ocl.synchronize(device)
 def memory():
  d=Path('/sys/bus/pci/devices')/(a.pci_device or ('0000:07:00.0' if device.index==0 else '0000:08:00.0'))
  return {k:int((d/k).read_text()) for k in ['mem_info_vram_used','mem_info_gtt_used'] if (d/k).is_file()}
 result=dict(family=a.family,batch=a.batch,device=a.device,checkpoint=str(checkpoint),
  checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest_path),input_shape=[a.batch,3,height,width],
  source_frame_shape=[display_h,width,3],source_frames='OpenCV-resized fixed wildlife fixtures; not native HD accuracy/video-decode qualification',
  precision='FP32',postprocess='NMS-free one-to-one confidence filtering' if a.family=='yolo26' else 'CPU NMS',
  raw_only=a.raw_only,correctness=[],timings={},memory_samples=[])
 result['queue_config']=dict(deferred_release_mib=int(os.environ.get('OPENCL_DEFERRED_RELEASE_MB','0')),async_op_limit=int(os.environ.get('OPENCL_ASYNC_OP_LIMIT','0')))
 result['postprocess_device']=a.postprocess_device
 if a.family=='yolo26' and a.postprocess_device=='cpu':
  result['postprocess']='Explicit CPU end-to-end top-k and confidence filtering; dense decoded boxes read back from GPU'
 with torch.inference_mode():
  cpu_inputs=[prepare(frame,'cpu').unsqueeze(0) for frame in frames]
  expected=[predict(reference,t).detach() for t in cpu_inputs]
  # Verify the selected batch shape, not just batch1 kernel selection.
  batch_cpu=torch.cat([cpu_inputs[i%len(frames)] for i in range(a.batch)],0)
  expected_batch=torch.cat([expected[i%len(frames)] for i in range(a.batch)],0)
  gpu_input=(prepare_batch([frames[i%len(frames)] for i in range(a.batch)])
             if a.input_packing=='batch' else batch_cpu.to(device))
  if a.input_packing=='batch':
   legacy=torch.stack([prepare(frames[i%len(frames)],device) for i in range(a.batch)])
   comparison_cpu=batch_cpu
   if a.fuse_input_focus:
    legacy=torch.cat([legacy[:,:,::2,::2],legacy[:,:,1::2,::2],legacy[:,:,::2,1::2],legacy[:,:,1::2,1::2]],1)
    comparison_cpu=torch.cat([batch_cpu[:,:,::2,::2],batch_cpu[:,:,1::2,::2],batch_cpu[:,:,::2,1::2],batch_cpu[:,:,1::2,1::2]],1)
   torch.testing.assert_close(gpu_input.cpu(),legacy.cpu(),rtol=0,atol=0)
   del legacy
   # GPU reciprocal multiplication differs from CPU division by <=1 float32 ULP;
   # require exact equivalence to the existing GPU preprocessing above.
   torch.testing.assert_close(gpu_input.cpu(),comparison_cpu,rtol=1e-7,atol=6e-8)
   del comparison_cpu
   result['batch_input_matches_legacy_gpu_exactly']=True
  result['copy_optimizations']=a.copy_optimizations;result['input_packing']=a.input_packing
  result['depthwise_vector4']=a.depthwise_vector4
  result['depthwise_stride2']=a.depthwise_stride2
  result['depthwise_layout']=a.depthwise_layout
  result['small_k']=a.small_k;result['pointwise_reduce']=a.pointwise_reduce
  result['gemm_n']=a.gemm_n
  result['gemm_experiments']=a.gemm_experiments
  result['cpu_head_layout']=a.cpu_head_layout
  result['compact_winograd']=a.compact_winograd
  result['winograd_mode']=a.winograd_mode
  result['gemm_pad_lds']=a.gemm_pad_lds
  result['fused_conv_silu_blocks']=fused_blocks
  result['fused_sppf_blocks']=sppf_blocks
  result['fused_input_focus_blocks']=focus_blocks
  result['network_entry_shape']=list(gpu_input.shape)
  result['prefetch_input']=a.prefetch_input
  # Capture shapes during the untimed validation forward, then remove every
  # hook before profiling/timing. Fused modules bypass their child Conv2d.
  conv_shapes=[];shape_hooks=[]
  if a.profile_stages:
   def shape_hook(name):
    def capture(module,args,value):
     c=module if isinstance(module,torch.nn.Conv2d) else module.conv
     conv_shapes.append(dict(name=name,input_shape=list(args[0].shape),output_shape=list(value.shape),
                             cin=c.in_channels,cout=c.out_channels,kernel=list(c.kernel_size),
                             stride=list(c.stride),groups=c.groups))
    return capture
   for name,module in network.named_modules():
    if isinstance(module,torch.nn.Conv2d) or name in fused_blocks:
     shape_hooks.append(module.register_forward_hook(shape_hook(name)))
  try:validation_raw=predict(network,gpu_input)
  finally:
   for handle in shape_hooks:handle.remove()
  if a.profile_stages:result['convolution_shapes']=conv_shapes
  actual=validation_raw.cpu();sync()
  host_output=torch.empty(actual.shape,dtype=actual.dtype) if a.reuse_readback else None
  def readback_tensor(raw):
   if host_output is None:return raw.cpu()
   assert raw.shape==host_output.shape and raw.dtype==host_output.dtype
   assert raw.is_contiguous(),'Reusable readback requires contiguous output; select channels-first for YOLO26'
   if a.readback_chunk_mib:
    src=raw.view(-1);dst=host_output.view(-1)
    step=a.readback_chunk_mib*1024**2//raw.element_size()
    for offset in range(0,src.numel(),step):dst[offset:offset+step].copy_(src[offset:offset+step])
   else:host_output.copy_(raw)
   return host_output
  if host_output is not None:
   torch.testing.assert_close(readback_tensor(validation_raw),actual,rtol=0,atol=0)
   result['readback_matches_legacy_exactly']=True
  result['reuse_readback']=a.reuse_readback;result['readback_chunk_mib']=a.readback_chunk_mib
  result['gpu_detections']=a.gpu_detections
  if a.gpu_detections:
   from gpu_detection_postprocess import gpu_end2end,gpu_v8_nms,gpu_yolox_nms,assert_same_detection_rows
   result['postprocess']='GPU confidence filtering, top-k and box/class gather; only final rows and status read back'
   result['gpu_selection']=dict(capacity=4096,threshold=.25,max_det=300,maximum_candidates=0,returned_bytes_per_batch=0,overflow='reject',ties='lower anchor/class ID first')
   if a.family=='yolox':
    result['postprocess']='GPU objectness/best-class filtering; candidate-only readback; original CPU batched NMS'
    result['gpu_selection'].update(capacity='all original anchors',overflow='impossible by allocation',ties='original anchor order restored before CPU NMS',max_det='unlimited as original YOLOX')
   if a.family=='wildlife':
    result['postprocess']='GPU confidence/argmax/anchor compaction; candidate-only readback; CPU class-aware NMS'
    result['gpu_selection'].update(capacity='all original anchors',overflow='impossible by allocation',ties='original anchor order restored before CPU NMS')
  def gpu_finalize(raw):
   rows,stats=(gpu_v8_nms(raw) if a.family=='wildlife' else gpu_yolox_nms(raw) if a.family=='yolox' else gpu_end2end(raw))
   result['gpu_selection']['maximum_candidates']=max(result['gpu_selection']['maximum_candidates'],stats['maximum_candidates'])
   result['gpu_selection']['returned_bytes_per_batch']=stats['returned_bytes']
   result['gpu_selection']['maximum_returned_bytes_per_batch']=max(result['gpu_selection'].get('maximum_returned_bytes_per_batch',0),stats['returned_bytes'])
   return rows
  assert actual.shape==expected_batch.shape,(actual.shape,expected_batch.shape)
  assert torch.isfinite(actual).all()
  try:
   if a.family=='yolo26' and a.postprocess_device=='gpu':
    # Near-equal scores can swap the order of complete detections. Match all
    # 300 rows by class, geometry and confidence; do not discard low scores.
    reordered=0;box_error=0.;score_error=0.
    for gpu_rows,cpu_rows in zip(actual,expected_batch):
     used=torch.zeros(len(cpu_rows),dtype=torch.bool)
     for index,row in enumerate(gpu_rows):
      candidates=(cpu_rows[:,5]==row[5]) & ~used
      candidates &= torch.isclose(cpu_rows[:,4],row[4],rtol=3e-3,atol=1e-6)
      candidates &= torch.isclose(cpu_rows[:,:4],row[:4],rtol=2e-3,atol=2e-3).all(1)
      choices=candidates.nonzero().flatten();assert len(choices),'Unmatched NMS-free detection'
      match=int(choices[(cpu_rows[choices,:4]-row[:4]).abs().amax(1).argmin()])
      used[match]=True;reordered+=match!=index
      box_error=max(box_error,float((cpu_rows[match,:4]-row[:4]).abs().max()))
      score_error=max(score_error,float((cpu_rows[match,4]-row[4]).abs()))
    check=dict(state='passed',comparison='All NMS-free rows matched irrespective of near-tie ranking',
               reordered_rows=reordered,max_matched_box_error=box_error,max_matched_score_error=score_error)
   else:
    torch.testing.assert_close(actual,expected_batch,rtol=3e-3,atol=2e-3)
    check=dict(state='passed',raw_max_absolute_error=float((actual-expected_batch).abs().max()))
  except (AssertionError,RuntimeError):
   torch.save(dict(actual=actual,expected=expected_batch),a.output.with_suffix('.failure.pt'))
   raise
  result['correctness'].append(check)
  # Both CPU and GPU output pass through the same CPU decoder/NMS.
  expected_dets=nms(expected_batch);actual_dets=nms(actual)
  if a.gpu_detections:
   assert_same_detection_rows(gpu_finalize(validation_raw),actual_dets)
   result['gpu_selection']['initial_rows_match_cpu_exactly']=True
  del validation_raw
  post_samples=[]
  for _ in range(5):
   post_start=time.perf_counter();post_outputs=nms(actual);post_samples.append(time.perf_counter()-post_start);del post_outputs
  result['cpu_postprocess_ms_per_image']=float(np.median(post_samples)*1000/a.batch)
  counts=lambda rows:[0 if row is None else len(row) for row in rows]
  assert counts(expected_dets)==counts(actual_dets),(counts(expected_dets),counts(actual_dets))
  result['correctness'][0]['detection_counts']=counts(actual_dets)
  dense_checks=a.family in ('wildlife','yolox') or (a.family=='yolo26' and a.postprocess_device=='cpu')
  expected_after=expected_batch if dense_checks else None
  # If the fixture suite exceeds one batch, qualify every remaining image at
  # the same selected batch shape, wrapping only the final partial batch.
  if dense_checks:
   for offset in range(a.batch,len(frames),a.batch):
    indexes=[(offset+i)%len(frames) for i in range(a.batch)]
    extra_cpu=torch.cat([cpu_inputs[i] for i in indexes],0)
    extra_expected=torch.cat([expected[i] for i in indexes],0)
    extra_gpu=(prepare_batch([frames[i] for i in indexes]) if a.input_packing=='batch' else extra_cpu.to(device))
    extra_raw=predict(network,extra_gpu)
    extra_actual=extra_raw.cpu();sync()
    if a.gpu_detections:assert_same_detection_rows(gpu_finalize(extra_raw),nms(extra_actual))
    torch.testing.assert_close(extra_actual,extra_expected,rtol=3e-3,atol=2e-3)
    assert counts(nms(extra_actual))==counts(nms(extra_expected))
    result['correctness'].append(dict(state='passed',fixture_indexes=indexes,
        raw_max_absolute_error=float((extra_actual-extra_expected).abs().max())))
    del extra_cpu,extra_expected,extra_gpu,extra_actual,extra_raw
   result['fixture_images_checked']=len(frames)
  print('CPU PARITY PASS',a.family,a.batch,result['correctness'],flush=True)
  del actual,expected_batch,batch_cpu,expected,expected_dets,actual_dets
  if a.profile_stages:
   profile_dir=a.output.parent/(a.output.stem+'-stages')
   profile_dir.mkdir(parents=True,exist_ok=False)
   diagnostics=[]
   # Warm complete operations before collecting synchronized stage diagnostics.
   for _ in range(2):
    warm_input=prepare_batch([frames[i%len(frames)] for i in range(a.batch)])
    warm_raw=predict(network,warm_input);warm_host=None
    warm_dets=gpu_finalize(warm_raw) if a.gpu_detections else nms(readback_tensor(warm_raw));sync()
    del warm_input,warm_raw,warm_host,warm_dets
   for repeat in range(a.profile_stages):
    row=dict(repeat=repeat)
    selected=[frames[(repeat*a.batch+i)%len(frames)] for i in range(a.batch)]
    started=time.perf_counter();packed=pack_batch(selected)
    row['cpu_pack_ms']=(time.perf_counter()-started)*1000
    def capture(name,operation):
     path=profile_dir/f'r{repeat}-{name}.csv'
     sync()
     with torch.ocl.profile(device,str(path)):
      begin=time.perf_counter();value=operation();sync();elapsed=time.perf_counter()-begin
     events=list(csv.DictReader(path.open()))
     row[name]=dict(wall_ms=elapsed*1000,gpu_event_ms=sum(float(e['duraion(ms)']) for e in events),
                    event_count=len(events),trace=str(path))
     result['memory_samples'].append(memory())
     return value
    staged_input=capture('upload_convert_normalize',lambda:upload_batch(packed))
    staged_raw=capture('forward',lambda:predict(network,staged_input))
    row['output']=dict(shape=list(staged_raw.shape),strides=list(staged_raw.stride()),
                       contiguous=staged_raw.is_contiguous(),bytes=staged_raw.numel()*staged_raw.element_size())
    staged_host=None
    if a.gpu_detections:
     staged_dets=capture('gpu_selection_and_final_readback',lambda:gpu_finalize(staged_raw))
     row['cpu_postprocess_ms']=0
    else:
     staged_host=capture('readback',lambda:readback_tensor(staged_raw))
     begin=time.perf_counter();staged_dets=nms(staged_host)
     row['cpu_postprocess_ms']=(time.perf_counter()-begin)*1000
    row['detection_counts']=counts(staged_dets)
    begin=time.perf_counter();del packed,staged_input,staged_raw,staged_host,staged_dets;sync()
    row['cleanup_ms']=(time.perf_counter()-begin)*1000
    diagnostics.append(row);print('STAGE PROFILE',json.dumps(row),flush=True)
   result['stage_profile']=dict(repeats=diagnostics,
       note='Synchronized serial stage diagnostics with GPU event profiling; CPU packing is not overlapped here. These times are not the prefetched pipeline throughput and omit CSV serialization from wall measurements.')
  def resident():return predict(network,gpu_input)
  frame_counter=[0]
  pack_pool=ThreadPoolExecutor(max_workers=1) if a.prefetch_input else None
  pending=[None]
  def select_frames():
   selected=[frames[(frame_counter[0]+i)%len(frames)] for i in range(a.batch)]
   frame_counter[0]=(frame_counter[0]+a.batch)%len(frames)
   return selected
  def pipeline():
   if pack_pool:
    if pending[0] is None:pending[0]=pack_pool.submit(pack_batch,select_frames())
    packed=pending[0].result()
    pending[0]=pack_pool.submit(pack_batch,select_frames())
    tensor=upload_batch(packed)
   else:
    selected=select_frames()
    tensor=(prepare_batch(selected) if a.input_packing=='batch' else
            torch.stack([prepare(frame,device) for frame in selected]))
   raw=predict(network,tensor)
   if a.raw_only:return raw
   return gpu_finalize(raw) if a.gpu_detections else nms(readback_tensor(raw))
  modes=[('device_resident_forward',resident),('decoded_HD_frame_to_detections' if not a.raw_only else 'decoded_HD_frame_to_raw_output',pipeline)]
  if a.profile_only:modes=[]
  for label,operation in modes:
   for _ in range(a.warmup):
    value=operation();sync();del value
   if label==modes[-1][0] and a.ready:
    a.ready.write_text('ready\n')
    deadline=time.monotonic()+180
    while not a.start_file.exists():
     if time.monotonic()>deadline:raise RuntimeError('Pair barrier timed out')
     time.sleep(.02)
   samples=[];window_start=time.monotonic()
   target_seconds=(a.forward_seconds if label=='device_resident_forward' and a.forward_seconds is not None else a.seconds)
   while len(samples)<a.iterations or (target_seconds and time.monotonic()-window_start<target_seconds):
    sync();begin=time.perf_counter();value=operation();sync();elapsed=time.perf_counter()-begin;del value
    samples.append(elapsed);result['memory_samples'].append(memory())
   window_end=time.monotonic()
   row=dict(batches=len(samples),images=len(samples)*a.batch,seconds=sum(samples),
            window_start_monotonic=window_start,window_end_monotonic=window_end,wall_seconds=window_end-window_start,
            images_per_second=len(samples)*a.batch/sum(samples),mean_batch_seconds=float(np.mean(samples)),
            p95_batch_seconds=float(np.percentile(samples,95)))
   result['timings'][label]=row;print('BENCHMARK',label,json.dumps(row),flush=True)
  if pack_pool:pack_pool.shutdown(wait=True)
  if expected_after is not None:
   after_raw=predict(network,gpu_input)
   after=readback_tensor(after_raw);sync()
   if host_output is not None:
    torch.testing.assert_close(after,after_raw.cpu(),rtol=0,atol=0)
    result['readback_matches_legacy_after_timing']=True
   torch.testing.assert_close(after,expected_after,rtol=3e-3,atol=2e-3)
   assert counts(nms(after))==counts(nms(expected_after))
   if a.gpu_detections:
    assert_same_detection_rows(gpu_finalize(after_raw),nms(after))
    result['gpu_selection']['after_timing_rows_match_cpu_exactly']=True
   result['correctness_after_timing']=dict(state='passed',raw_max_absolute_error=float((after-expected_after).abs().max()))
   print('CPU PARITY AFTER TIMING PASS',flush=True)
  result['allocator']=torch.ocl.memory_stats(device)
  result['peak_sampled_vram_bytes']=max((r['mem_info_vram_used'] for r in result['memory_samples'] if 'mem_info_vram_used' in r),default=None)
  result['peak_sampled_gtt_bytes']=max((r['mem_info_gtt_used'] for r in result['memory_samples'] if 'mem_info_gtt_used' in r),default=None)
  result['correctness_passed']=True
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT',a.output,flush=True)

if __name__=='__main__':main()
