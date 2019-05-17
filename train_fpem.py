#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
  
  @author: pang jingcheng
  @file: train_fpem.py
  @time: 2019-03-19
'''


import argparse
from datetime import datetime

import sys
import os

import random

random.seed(7)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from rlpytorch import *

# in this main page we init some settings for game context and then run it!
if __name__ == '__main__':
    verbose = False

    # some modules necessary for game context
    trainer = Trainer(verbose=verbose)
    runner = SingleProcessRun()
    evaluator = Evaluator(stats=False, verbose=verbose)
    env, all_args = load_env(os.environ, trainer=trainer, runner=runner, evaluator=evaluator)

    GC = env["game"].initialize_fpem()

    model = env["model_loaders"][0].load_model(GC.params)
    env["mi"].add_model("model", model, opt=True)
    env["mi"].add_model("actor", model, copy=True, cuda=all_args.gpu is not None, gpu_id=all_args.gpu)

    trainer.setup(sampler=env["sampler"], mi=env["mi"], rl_method=env["method"])
    evaluator.setup(sampler=env["sampler"], mi=env["mi"].clone(gpu=all_args.gpu))

    # those callback function will be called in order when batch are filled up
    if not all_args.actor_only:
        GC.reg_callback("train1", trainer.train)
    GC.reg_callback("actor1", trainer.actor)
    GC.reg_callback("actor0", evaluator.actor)

    # a function for log printing
    def summary(i, cur_policy):
        trainer.episode_summary(i, cur_policy)
        evaluator.episode_summary(i)

    def start(i):
        trainer.episode_start(i)
        evaluator.episode_start(i)

    runner.setup(GC, episode_summary=summary, episode_start=start)

    # main process start running!!!!
    runner.fpem_run()

