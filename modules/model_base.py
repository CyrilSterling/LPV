# import logging
import copy
import torch.nn as nn
import torch

from modules.attention import PositionAttention
from modules.svtr_backbone import svtr_tiny, svtr_small, svtr_base
from modules.layers import Trans

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class BaseModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.name = opt.exp_name
        self.out_channels = opt.d_model

        self.backbone = eval(opt.backbone)(pretrained=False)
        pm_stride = self.backbone.pm_stride
        opt.fH = opt.imgH // (4 * pm_stride[0][0] * pm_stride[1][0])
        opt.fW = opt.imgW // (4 * pm_stride[0][1] * pm_stride[1][1])
        
        mode = opt.attention_mode if opt.attention_mode else 'nearest'

        attn_layer = PositionAttention(max_length=opt.max_length + 2,  mode=mode, in_channels=self.out_channels, num_channels=self.out_channels//8, h=opt.fH, w=opt.fW)
        trans_layer = Trans(opt)
        cls_layer = nn.Linear(self.out_channels, len(opt.character) + 2)

        self.attention = _get_clones(attn_layer, opt.attn)
        self.trans = _get_clones(trans_layer, opt.attn-1)
        self.cls = _get_clones(cls_layer, opt.attn)


    def forward(self, images, is_eval=False):
        n,_,h,w = images.shape
        features = self.backbone(images)  # (N, E, H, W)
        features = features.permute(0,2,1).reshape(n, -1, self.opt.fH, self.opt.fW)

        attn_vecs, attn_scores, attn_scores_map = None, None, None
        if self.opt.no_debug:
            attn_vecs, attn_scores_map = self.attention[0](features, attn_vecs)
            for i in range(1, len(self.attention)):
                features = self.trans[i-1](features, attn_scores_map, use_mask=self.opt.mask, is_eval=is_eval)
                attn_vecs, attn_scores_map = self.attention[i](features, attn_vecs)  # (N, T, E), (N, T, H, W)
            return self.cls[2](attn_vecs)
        else:
            all_vecs = []
            all_scores = []
            logits = []
            pt_lengths = []
            attns = []
            masks = []

            attn_vecs, attn_scores_map = self.attention[0](features, attn_vecs)
            all_vecs.append(attn_vecs)
            all_scores.append(attn_scores_map)
            logit = self.cls[0](all_vecs[0]) # (N, T, C)
            logits.append(logit)
            for i in range(1, len(self.attention)):
                use_mask = self.opt.mask
                features, mask, attn = self.trans[i-1](features, attn_scores_map, use_mask=use_mask, is_eval=is_eval)
                
                attn_vecs, attn_scores_map = self.attention[i](features, attn_vecs)  # (N, T, E), (N, T, H, W)
                
                all_vecs.append(attn_vecs)
                all_scores.append(attn_scores_map)
                if is_eval:
                    attns.append(attn)
                    masks.append(mask)
            
                logit = self.cls[i](all_vecs[-1]) # (N, T, C)
                logits.append(logit)
        
            if is_eval:
                attns = torch.stack(attns, dim=0).permute(2,0,1,3,4) # N, each trans, each layer, HW, HW
                return all_scores, logits, attns
            else:
                return torch.cat(logits, dim=0)
