#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

  @author: pang jingcheng
  @file: fpem.py
  @time: 2019-03-19
'''

import os
import torch
import random
import numpy as np

random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)

from rlpytorch import *


if __name__ == '__main__':

    # environment settings
    verbose = False

    # some modules necessary for game context
    trainer = Trainer(verbose=verbose)
    runner = SingleProcessRun()
    evaluator = Evaluator(stats=False, verbose=verbose)
    env, all_args = load_env(os.environ, trainer=trainer, runner=runner, evaluator=evaluator)

    GC = env["game"].initialize_selfplay()

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
    runner.run()

