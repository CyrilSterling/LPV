import torch
import torch.nn as nn

from modules.transformer import (PositionalEncoding,
                                 TransformerEncoder,
                                 TransformerEncoderLayer,
                                 TransformerDecoder,
                                 TransformerDecoderLayer)

class Trans(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.d_model = opt.d_model
        self.opt = opt
        nhead = opt.nhead
        d_inner = opt.d_inner
        dropout = opt.dropout
        activation = opt.activation
        num_layers = opt.trans_ln

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=opt.fH*opt.fW)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, 
                dim_feedforward=d_inner, dropout=dropout, activation=activation, debug=(not opt.no_debug))
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, feature, attn_map=None, use_mask=False, debug=False, is_eval=False):
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)

        if use_mask:
            _,t,h,w = attn_map.shape
            location_mask = (attn_map.view(n, t, -1).permute(0, 2, 1) > 0.05).type(torch.float)  # n,s,t
            location_mask = location_mask.bmm(location_mask.permute(0,2,1))
            location_mask = location_mask.new_zeros((h*w, h*w)).masked_fill(location_mask>0, float('-inf'))
            location_mask = location_mask.repeat(8,1,1,1).permute(1,0,2,3).flatten(0,1)
        else:
            location_mask = None

        feature = self.pos_encoder(feature)
        if self.opt.no_debug or not is_eval:
            feature = self.transformer(feature, mask=location_mask)
            feature = feature.permute(1, 2, 0).view(n, c, h, w)
            return feature, None, None
        else:
            feature, attns = self.transformer(feature, mask=location_mask, debug=True)
            feature = feature.permute(1, 2, 0).view(n, c, h, w)
            return feature, location_mask, attns