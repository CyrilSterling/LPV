"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

from modules.model_base import BaseModel

import math

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        
        if opt.backbone == 'svtr_base':
            opt.imgH, opt.imgW = 48, 160
            opt.d_model, opt.d_inner = 384, 1536
        self.str_model = BaseModel(opt)

    def forward(self, input, is_eval=False):
        prediction = self.str_model(input, is_eval=is_eval)
        return prediction


