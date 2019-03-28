#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
  
  @author: pang jingcheng
  @file: fpem_model.py
  @time: 2019-03
'''

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from collections import Counter

import sys
sys.path.append('../../')

from rlpytorch import Model, ActorCritic
from rts.game_MC.actor_critic_changed import ActorCriticChanged
from rts.game_MC.forward_predict import ForwardPredict
from rts.game_MC.trunk import MiniRTSNet

class Model_ActorCritic(Model):
    def __init__(self, args):
        super(Model_ActorCritic, self).__init__(args)
        self._init(args)

    def _init(self, args):

        # Then we load 10 bps' paras
        params = args.params
        assert isinstance(params["num_action"], int), "num_action has to be a number. action = " + str(params["num_action"])
        self.params = params

        self.bp_list = []

        self.PNet = MiniRTSNet(args)
        model_dict = self.PNet.state_dict()
        pretrain_dict = torch.load('fpem/fpeminit.bin')

        # to load pretrained model parameters
        # tested to be useful
        pretrained_dict = {}
        for k,v in pretrain_dict["stats_dict"].items():
            if 'net.' in k:
                k = k[4:]

            if k in model_dict.keys():
                pretrained_dict[k] = v

        model_dict.update(pretrained_dict)
        self.PNet.load_state_dict(model_dict)

        last_num_channel = self.PNet.num_channels[-1]

        if self.params.get("model_no_spatial", False):
            self.num_unit = params["num_unit_type"]
            linear_in_dim = last_num_channel
        else:
            linear_in_dim = last_num_channel * 25

        self.linear_in_dim = linear_in_dim

        # represent for bp/weight/value function respectively
        for i in range(10):
            self.bp_list.append(nn.Linear(linear_in_dim, params["num_action"]).cuda())

            cur_dict = {}
            form_dict = self.bp_list[i].state_dict()
            for k, v in pretrain_dict["stats_dict"].items():
                for k_cur, v_cur in form_dict.items():
                    if k in k_cur:
                        cur_dict[k_cur] = v
                        break

            form_dict.update(cur_dict)
            self.bp_list[i].load_state_dict(form_dict)

        self.linear_weight = nn.Linear(linear_in_dim, 10)

        self.linear_value = nn.Linear(linear_in_dim, 1)
        cur_dict = {}
        form_dict = self.linear_value.state_dict()
        for k, v in pretrain_dict["stats_dict"].items():
            for k_cur, v_cur in form_dict.items():
                if k in k_cur:
                    cur_dict[k_cur] = v
                    break

        form_dict.update(cur_dict)
        self.linear_value.load_state_dict(form_dict)

        self.relu = nn.LeakyReLU(0.1)

        self.Wt = nn.Linear(linear_in_dim + params["num_action"], linear_in_dim)
        self.Wt2 = nn.Linear(linear_in_dim, linear_in_dim)
        self.Wt3 = nn.Linear(linear_in_dim, linear_in_dim)

        self.softmax = nn.Softmax()

    def get_define_args():
        return MiniRTSNet.get_define_args()

    # Run the policy net!
    def forward(self, x, cur_policy=9):
        if self.params.get("model_no_spatial", False):
            weight_prob = None
            # Replace a complicated network with a simple retraction.
            # Input: batchsize, channel, height, width
            xreduced = x["s"].sum(2).sum(3).squeeze()
            xreduced[:, self.num_unit:] /= 20 * 20

            output = self._var(xreduced)
        else:
            output = self.PNet(self._var(x["s"]))

        return self.decision(output, cur_policy)

    def decision(self, h, cur_policy=9):

        h = self._var(h)

        sample_size = h.size(0)

        policy = np.zeros([sample_size, self.params["num_action"]], dtype=np.float32)

        weight_prob = np.zeros([sample_size], dtype=np.float32)
        bp_selected = np.zeros([sample_size], dtype=np.int32)

        weight = self.softmax(self.linear_weight(h)[:, :(cur_policy+1)])
        weight_argmax = torch.argmax(weight, dim=1).detach().cpu().numpy()
        weight_numpy = weight.detach().cpu().numpy()

        for index in range(sample_size):
            policy_index = weight_argmax[index]
            weight_prob[index] = weight_numpy[index, policy_index]
            bp_selected[index] = policy_index

            policy[index, :] = self.bp_list[policy_index](h[index]).detach().cpu().numpy()

        policy = self.softmax(torch.from_numpy(policy))
        bp_selected = torch.from_numpy(bp_selected)
        value = self.linear_value(h)
        decion_result = dict(h=h, V=value, pi=policy, weight_pi=weight_prob, bp=bp_selected, action_type=0)

        return decion_result

    def decision_fix_weight(self, h):
        # Copy linear policy and linear value
        if not hasattr(self, "fixed"):
            self.fixed = dict()
            self.fixed["linear_policy"] = deepcopy(self.linear_policy)
            self.fixed["linear_value"] = deepcopy(self.linear_value)

        policy = self.softmax(self.fixed["linear_policy"](h))
        value = self.fixed["linear_value"](h)
        return dict(h=h, V=value, pi=policy,  action_type=0)

    def transition(self, h, a):
        ''' A transition model that could predict the future given the current state and its action '''
        h = self._var(h)
        na = self.params["num_action"]
        a_onehot = h.data.clone().resize_(a.size(0), na).zero_()
        a_onehot.scatter_(1, a.view(-1, 1), 1)
        input = torch.cat((h, self._var(a_onehot)), 1)

        h2 = self.relu(self.Wt(input))
        h3 = self.relu(self.Wt2(h2))
        h4 = self.relu(self.Wt3(h3))

        return dict(hf=h4)

    def reset_forward(self):
        self.Wt.reset_parameters()
        self.Wt2.reset_parameters()
        self.Wt3.reset_parameters()

# Format: key, [model, method]
# if method is None, fall back to default mapping from key to method
Models = {
    "actor_critic": [Model_ActorCritic, ActorCritic],
    "actor_critic_changed": [Model_ActorCritic, ActorCriticChanged],
    "forward_predict": [Model_ActorCritic, ForwardPredict]
}
