import os
import pickle
import torch
import numpy as np

N_SHAPE=100


'''

Default Input Format : .pkl dictionary, consists of ['full_pose','shape','exp','cam']

full_pose : N * joints * 3 (or 3*3)
shape : 100
exp : N * coeff
cam : N * 3 (for the orthographic)

'''

def load_smplx(filepath='examples/smplx.pkl'):
    # load pixie animation poses
    assert os.path.exists(filepath), f'{filepath} does not exist'
    with open(filepath, 'rb') as f:
        codedict = pickle.load(f)

    ## KEY #1 : full_pose

    assert 'full_pose' in codedict.keys()
    full_pose = torch.from_numpy(codedict['full_pose'])

    n_frames = len(full_pose)
    device = full_pose.device

    ## KEY #2 : cam

    if 'cam' in codedict.keys():
        cam = torch.from_numpy(codedict['cam']).to(device)
    else:
        print('cam not in {}. Automatically fill with 0.'.format(filepath))
        cam = torch.zeros((n_frames, 3)).to(device)
    assert len(cam) == n_frames, 'length of cam is {}, while number of frames is {}'.format(len(cam, n_frames))

    ## KEY #3 : exp

    if 'exp' in codedict.keys():
        exp = torch.from_numpy(codedict['exp']).to(device)
    else:
        print('exp not in {}. Automatically fill with 0.'.format(filepath))
        exp = torch.zeros((n_frames, 100)).to(device)
    assert len(exp) == n_frames, 'length of exp is {}, while number of frames is {}'.format(len(exp, n_frames))

    ## KEY #4 : shape

    if 'shape' in codedict.keys():
        shape = torch.from_numpy(codedict['shape']).to(device)
    else:
        print('shape not in {}. Automatically fill with 0.'.format(filepath))
        shape = torch.zeros((100)).to(device)

    return full_pose, cam, exp, shape

def load_smpl(filepath=''):
    raise NotImplementedError

def load_flame(filepath=''):
    raise NotImplementedError