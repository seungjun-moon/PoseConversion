import os
import pickle
import torch
import numpy as np

def load_pixie_smplx(actions_file='examples/pixie_radioactive.pkl'):
    # load pixie animation poses
    assert os.path.exists(actions_file), f'{actions_file} does not exist'
    with open(actions_file, 'rb') as f:
        codedict = pickle.load(f)
    full_pose = torch.from_numpy(codedict['full_pose'])
    cam = codedict['cam']
    exp = codedict['exp']
    return full_pose, exp, cam

def load_deca_flame(actions_file=''):
    raise NotImplementedError

