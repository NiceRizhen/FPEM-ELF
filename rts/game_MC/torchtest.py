#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
  
  @author: pang jingcheng
  @file: torchtest.py
  @time: 2019-03
'''

import torch
import numpy as np
import torch.nn as nn

from rts.game_MC.trunk import MiniRTSNet


a = torch.load('smv1model.bin')
#print(a["stats_dict"])

for key in a["stats_dict"]:
    print(key)