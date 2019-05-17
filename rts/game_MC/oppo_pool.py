#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
  Opponent pool models.

  @author: pang jingcheng
  @file: opponent_pool.py
  @time: 2019-03
'''

import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import random
from copy import deepcopy
from collections import Counter
from torch.autograd import Variable
from rlpytorch import Model
from rlpytorch.methods.oppo import OPPO
from rts.game_MC.actor_critic_changed import ActorCriticChanged
from rts.game_MC.forward_predict import ForwardPredict
from rts.game_MC.trunk import MiniRTSNet

class oppo_poolModel(Model):
    def __init__(self, args):
        super(oppo_poolModel, self).__init__(args)
        self._init(args)

    def _init(self, args):

        params = args.params
        assert isinstance(params["num_action"], int), "num_action has to be a number. action = " + str(
            params["num_action"])
        self.params = params

        self.Net = MiniRTSNet(args)

        last_num_channel = self.Net.num_channels[-1]

        if self.params.get("model_no_spatial", False):
            self.num_unit = params["num_unit_type"]
            linear_in_dim = last_num_channel
        else:
            linear_in_dim = last_num_channel * 25

        self.linear_in_dim = linear_in_dim

        # for base policy
        self.bp_list = [nn.Linear(linear_in_dim, params["num_action"]) for i in range(10)]
        for i, bp in enumerate(self.bp_list):
            setattr(self, "bp%d" % (i + 1), bp)

        self.value_list = [nn.Linear(linear_in_dim, 1) for i in range(10)]
        for i, val in enumerate(self.value_list):
            setattr(self, "value%d" % (i + 1), val)

        self.relu = nn.LeakyReLU(0.1)

        self.softmax = nn.Softmax(dim=1)

    def get_define_args():
        return MiniRTSNet.get_define_args()

    # Run the policy net!
    def forward(self, x, cur_policy=None):

        cp = cur_policy["cur_policy"]

        # it's policy index for this game
        p_index = cur_policy.get("p_index", random.randint(0, cp))

        state_feature = self.Net(self._var(x["s"]))

        if not self.volatile:
            # if it's time for training, we use bp[cur_policy] and update it
            policy = self.bp_list[cp](state_feature)
        else:
            policy = self.bp_list[p_index](state_feature)

        policy = self.softmax(policy)
        value = self.value_list[cp](state_feature)

        decion_result = dict(h=state_feature, V=value, pi=policy, action_type=0)

        return decion_result

# Format: key, [model, method]
# if method is None, fall back to default mapping from key to method
Models = {
    "actor_critic": [oppo_poolModel, OPPO],
    "actor_critic_changed": [oppo_poolModel, ActorCriticChanged],
    "forward_predict": [oppo_poolModel, ForwardPredict]
}
