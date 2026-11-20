"""NMS-free GPU selection; CPU reads status and final rows, never dense scores."""
import torch

def gpu_end2end(predictions,threshold=.25,maxdet=300,capacity=4096):
    out,status=torch.ops.pytorch_ocl.gpu_detections(predictions,threshold,maxdet,capacity)
    status_cpu=status.cpu()
    if status_cpu.numel() and bool(status_cpu[:,1].any()):
        raise RuntimeError('GPU detector received nonfinite input')
    maximum=int(status_cpu[:,0].max()) if len(status_cpu) else 0
    if maximum>capacity:
        raise RuntimeError(f'GPU detector candidate budget exceeded: {maximum}>{capacity}; output rejected')
    host=out.cpu()
    if not bool(torch.isfinite(host).all()):raise RuntimeError('Nonfinite GPU detections')
    counts=[min(int(n),host.shape[1]) for n in status_cpu[:,0]]
    return [host[i,:n] for i,n in enumerate(counts)],dict(
        candidates=[int(n) for n in status_cpu[:,0]],maximum_candidates=maximum,
        returned_bytes=host.numel()*host.element_size()+status_cpu.numel()*status_cpu.element_size())

def assert_same_detection_rows(actual,expected):
    assert len(actual)==len(expected)
    for got,want in zip(actual,expected):
        if got is None or want is None:
            assert got is None and want is None
            continue
        # All values are copied, not recomputed; only rank-tie ordering may differ.
        assert sorted(tuple(r) for r in got.tolist())==sorted(tuple(r) for r in want.tolist())

def gpu_v8_nms(predictions,threshold=.25,iou=.7,maxdet=300):
    """Single-best-class Detect: GPU compaction, original-order CPU NMS.

    Capacity equals all anchors: no candidate cap or top-k before native NMS.
    """
    from torchvision.ops import nms
    from ultralytics.utils.ops import xywh2xyxy
    out,status=torch.ops.pytorch_ocl.gpu_v8_candidates(predictions,threshold)
    status_cpu=status.cpu()
    if status_cpu.numel() and bool(status_cpu[:,1].any()):
        raise RuntimeError('GPU detector received nonfinite input')
    counts=[int(n) for n in status_cpu[:,0]]
    maximum=max(counts,default=0)
    # Only the candidate prefix crosses PCIe, including padding up to this
    # batch's maximum count. The full-capacity buffer remains on the GPU.
    host=out[:,:maximum,:].contiguous().cpu() if maximum else torch.empty(len(counts),0,7)
    results=[]
    for i,count in enumerate(counts):
        rows=host[i,:count]
        if not count:
            results.append(torch.empty(0,6));continue
        if not bool(torch.isfinite(rows).all()):raise RuntimeError('Nonfinite GPU candidates')
        # Restore original anchor order, including NMS's equal-score ordering.
        rows=rows[rows[:,6].argsort()]
        detections=torch.cat((xywh2xyxy(rows[:,:4]),rows[:,4:6]),1)
        if count>30000:detections=detections[detections[:,4].argsort(descending=True)[:30000]]
        keep=nms(detections[:,:4]+detections[:,5:6]*7680,detections[:,4],iou)[:maxdet]
        results.append(detections[keep])
    return results,dict(candidates=counts,maximum_candidates=maximum,
        returned_bytes=host.numel()*host.element_size()+status_cpu.numel()*status_cpu.element_size())

def gpu_yolox_nms(predictions,threshold=.25,iou=.7):
    """YOLOX best class/objectness filtering; original batched CPU NMS."""
    from torchvision.ops import batched_nms
    out,status=torch.ops.pytorch_ocl.gpu_yolox_candidates(predictions,threshold)
    status_cpu=status.cpu()
    if status_cpu.numel() and bool(status_cpu[:,1].any()):raise RuntimeError('GPU detector received nonfinite input')
    counts=[int(n) for n in status_cpu[:,0]];maximum=max(counts,default=0)
    host=out[:,:maximum,:].contiguous().cpu() if maximum else torch.empty(len(counts),0,8)
    results=[]
    for i,count in enumerate(counts):
        if not count:results.append(None);continue
        rows=host[i,:count]
        if not bool(torch.isfinite(rows).all()):raise RuntimeError('Nonfinite GPU candidates')
        rows=rows[rows[:,7].argsort()]
        detections=rows[:,:7].clone()
        detections[:,0]=rows[:,0]-rows[:,2]/2;detections[:,1]=rows[:,1]-rows[:,3]/2
        detections[:,2]=rows[:,0]+rows[:,2]/2;detections[:,3]=rows[:,1]+rows[:,3]/2
        keep=batched_nms(detections[:,:4],detections[:,4]*detections[:,5],detections[:,6],iou)
        results.append(detections[keep])
    return results,dict(candidates=counts,maximum_candidates=maximum,
        returned_bytes=host.numel()*host.element_size()+status_cpu.numel()*status_cpu.element_size())
