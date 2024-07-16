# -*- coding: utf-8 -*-
"""
author: shanzha
WeChat: shanzhan09
create_time: 2021/12/30 14:16
"""
import torch
from newModel import ModelBase
from swinunet import SwinTransformerSys
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = ModelBase(num_classes=2)


model.cuda()

print(model)

x = torch.rand(8, 3, 224, 224).to(device)
pred = model(x)
print(pred.shape)
