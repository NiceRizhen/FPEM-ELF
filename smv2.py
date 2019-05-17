'''
  Update new model's params with pre-trained model.

  @python version : 3.6.4
  @author : pangjc
  @time : 2019/4/2
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
from rlpytorch.model_loader import load_2models_env

if __name__ == '__main__':
    verbose = False

    trainer1 = Trainer(verbose=verbose, _method_name='single_model', _save_dir='./model/sm/smv2', _save_prefix='single')
    trainer2 = Trainer(verbose=verbose, _method_name='oppo_pool', _save_dir='./model/oppo_pool/smv2', _save_prefix='oppo')
    runner = SingleProcessRun()

    env, all_args = load_2models_env(os.environ, trainer1=trainer1, trainer2=trainer2, runner=runner)

    GC = env["game"].initialize_smv2()

    model1 = env["model_loaders"][0].load_model(GC.params)
    env["mi1"].add_model("model", model1, opt=True)
    env["mi1"].add_model("actor", model1, copy=True, cuda=all_args.gpu is not None, gpu_id=all_args.gpu)

    model2 = env["model_loaders"][1].load_model(GC.params)
    env["mi2"].add_model("model", model2, opt=True)
    env["mi2"].add_model("actor", model2, copy=True, cuda=all_args.gpu is not None, gpu_id=all_args.gpu)

    trainer1.setup(sampler=env["sampler1"], mi=env["mi1"], rl_method=env["method1"])
    trainer2.setup(sampler=env["sampler2"], mi=env["mi2"], rl_method=env["method2"])

    GC.reg_callback("train1", trainer1.train)
    GC.reg_callback("train2", trainer2.train)
    GC.reg_callback("actor1", trainer1.actor)
    GC.reg_callback("actor2", trainer2.actor)

    def summary(i, cur_policy=None):
        trainer1.episode_summary(i, cur_policy)
        trainer2.episode_summary(i, cur_policy)

    def start(i):
        trainer1.episode_start(i)
        trainer2.episode_start(i)

    runner.setup(GC, episode_summary=summary, episode_start=start)
    runner.run()
