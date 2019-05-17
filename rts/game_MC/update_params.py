'''
  Update new model's params with pre-trained model.

  @python version : 3.6.4
  @author : Pang Jingcheng
  @time : 2019/4/27
'''


import sys
sys.path.append('../../')

import torch
from rts.game_MC.model import Model_ActorCritic
from rts.game_MC.fpem_model import FPEM_Model

class par():
    def __init__(self):
        self.params = {'map_x': 20, 'map_y': 20, 'max_unit_cmd': 1, 'num_action': 9, 'num_cmd_type': 4, 'num_planes': 22,
                      'num_planes_per_time_stamp': 22, 'num_unit_type': 6, 'reduced_dim': 550, 'resource_dim': 10,
                      'rts_engine_version': '1f790173095cd910976d9f651b80beb872ec5d12_staged',
                      'players': [{'type': 'AI_NN', 'fs': '50', 'args': 'backup/AI_SIMPLE|start/500|decay/0.99'},
                                  {'type': 'AI_SIMPLE', 'fs': '20'}], 'num_group': 2, 'action_batchsize': 128,
                      'train_batchsize': 128, 'T': 20, 'model_no_spatial': False}

        self.arch = "ccpccp;-,64,64,64,-"


if __name__ == "__main__":
    args = par()

    # step1 : load model
    # pre-trained model
    source = Model_ActorCritic(args)
    source.load('/home/amax/Desktop/elfbackup/ELF/smv1/selfplay-3.bin')

    # new model
    target = FPEM_Model(args)
    target.load('/home/amax/Desktop/elfbackup/ELF/fpeminit.bin')


    # step2 : update parameters
    s_params = source.state_dict()
    t_params = target.state_dict()

    for key in target.state_dict().keys():
        if key == 'bp6.bias':
            print(target.state_dict()[key])

    for key in t_params.keys():
        if 'Net' in key:
            t_params[key] = s_params[key.replace("CNNNet",'net')]
            print('changing %s'%key)

        elif 'bp' in key and 'weight' in key:
            t_params[key] = s_params['linear_policy.weight']
            print('changing %s' % key)

        elif 'bp' in key and 'bias' in key:
            t_params[key] = s_params['linear_policy.bias']
            print('changing %s' % key)

        elif 'value' in key and 'weight' in key:
            t_params[key] = s_params['linear_value.weight']
            print('changing %s' % key)

        elif 'value' in key and 'bias' in key:
            t_params[key] = s_params['linear_value.bias']
            print('changing %s' % key)


    # step3 : save new model(with pretrain parameters)
    target.load_state_dict(t_params)
    for key in target.state_dict().keys():
        if 'bp' in key and 'bias' in key:
            print(target.state_dict()[key])

    target.save('fpeminit.bin')
