#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
  Fictitious Play with Expanding Models based on pytorch.

  @author: pangjc
  @file: fpem_model.py
  @time: 2019-03
'''

import torch
import torch.nn as nn

import sys

sys.path.append('../../')

from rlpytorch import Model
from torch.autograd import Variable
from rlpytorch.methods.fpem import FPEM
from rlpytorch.sampler.FPEMsampler import FPEMSampler
from rts.game_MC.trunk import MiniRTSWeight


class FPEM_Model(Model):
    def __init__(self, args):
        super(FPEM_Model, self).__init__(args)
        self._init(args)

    def _init(self, args):

        # Then we load 10 bps' paras
        params = args.params
        assert isinstance(params["num_action"], int), "num_action has to be a number. action = " + str(
            params["num_action"])
        self.params = params

        self.CNNNet = MiniRTSWeight(args)

        last_num_channel = self.CNNNet.num_channels[-1]

        if self.params.get("model_no_spatial", False):
            self.num_unit = params["num_unit_type"]
            linear_in_dim = last_num_channel
        else:
            linear_in_dim = last_num_channel * 25

        # for pi selector
        self.weight = nn.Linear(linear_in_dim, 10)

        # for base policy
        self.bp_list = [nn.Linear(linear_in_dim, params["num_action"]) for i in range(10)]
        for i, bp in enumerate(self.bp_list):
            setattr(self, "bp%d" % (i + 1), bp)

        self.value_list = [nn.Linear(linear_in_dim, 1) for i in range(10)]
        for i, val in enumerate(self.value_list):
            setattr(self, "value%d" % (i + 1), val)

        self.relu = nn.LeakyReLU(0.1)

        self.Wt = nn.Linear(linear_in_dim + params["num_action"], linear_in_dim)
        self.Wt2 = nn.Linear(linear_in_dim, linear_in_dim)
        self.Wt3 = nn.Linear(linear_in_dim, linear_in_dim)

        self.softmax = nn.Softmax(dim=-1)

    def get_define_args():
        return MiniRTSWeight.get_define_args()

    # replay:{
    #     "h" features processed by cnn;
    #     "V" Value;
    #     "pi" probs calculated by base policy;
    #     "s_pi" probs calculated by pi_selector;
    # }
    def forward(self, x, cur_policy=None):

        cp = cur_policy["cur_policy"]

        state = self._var(x["s"])

        sample_size = state.size(0)
        state_feature = self.CNNNet(state)

        # to make sure policy selector get policy between 0-cur_policy
        selector_output = Variable(torch.zeros((sample_size, 10)), volatile=self.volatile).cuda()
        selector_output[:, 0:cp + 1] = self.softmax(self.weight(state_feature))[:, 0:cp + 1]

        # it's not training
        if self.volatile:
            _, max_pi_index = selector_output.max(dim=-1)
            policy = Variable(torch.zeros((sample_size, self.params["num_action"])), volatile=self.volatile).cuda()
            for i in range(sample_size):
                policy[i, :] = self.bp_list[max_pi_index[i].data[0]](state_feature[i])

        # else if it's training time
        else:
            policy = self.bp_list[cp](state_feature)

        policy = self.softmax(policy)
        value = self.value_list[cp](state_feature)
        decion_result = dict(h=state_feature, V=value, pi=policy, s_pi=selector_output, action_type=0)

        return decion_result

    def reset_forward(self):
        self.Wt.reset_parameters()
        self.Wt2.reset_parameters()
        self.Wt3.reset_parameters()


# Format: key, [model, method, (sampler)]
# if method is None, fall back to default mapping from key to method
Models = {
    "actor_critic": [FPEM_Model, FPEM, FPEMSampler]
}
