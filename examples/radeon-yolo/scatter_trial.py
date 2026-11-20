"""Build assignment masks directly for indices produced by max and top-k."""
import os
from types import MethodType
import torch

def install(model):
    for branch in (model.criterion.one2many,model.criterion.one2one):
        assigner=branch.assigner
        original=assigner.select_highest_overlaps
        def wrap(original):
            @torch.no_grad()
            def select(self, mask_pos, overlaps, n_max_boxes, align_metric):
                if os.environ.get('PROFILE_DIRECT_MASKS') != '1' or mask_pos.device.type != 'ocl':
                    return original(mask_pos,overlaps,n_max_boxes,align_metric)
                multi=(mask_pos.sum(-2).unsqueeze(1)>1).expand(-1,n_max_boxes,-1)
                winners=overlaps.max(1).indices.unsqueeze(1)
                boxes=torch.arange(n_max_boxes,device=mask_pos.device).reshape(1,-1,1)
                selected=(boxes==winners).to(mask_pos.dtype)
                mask_pos=torch.where(multi,selected,mask_pos)
                if self.topk2!=self.topk:
                    if self.topk2==1:
                        best=torch.topk(align_metric*mask_pos,1,dim=-1).indices
                        anchors=torch.arange(mask_pos.shape[-1],device=mask_pos.device)
                        mask_pos*= (anchors==best).to(mask_pos.dtype)
                    else:
                        ids=torch.topk(align_metric*mask_pos,self.topk2,dim=-1).indices
                        mask_pos*=torch.zeros_like(mask_pos).scatter_(-1,ids,1)
                fg,indices=mask_pos.to(torch.int32).max(-2)
                return indices,fg.to(mask_pos.dtype),mask_pos
            return select
        assigner.select_highest_overlaps=MethodType(wrap(original),assigner)
