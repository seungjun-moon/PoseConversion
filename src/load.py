import os
import json
import pickle
import torch
import numpy as np


def load_pickle(filepath):

    '''
    Default Input Format : .pkl dictionary, consists of ['full_pose','shape','exp','cam']

    full_pose : N * joints * 3 (or 3*3)
    shape : 100
    exp : N * coeff
    cam : N * 3 (for the orthographic)
    '''

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

def load_json(blendshape_path):
    '''
    Input: .json dictionary path, only for the BlendShape
    Output: N * 52
    '''
    f = open(blendshape_path)
    blendshape = json.load(f)
    coeffs = torch.from_numpy(np.asarray(blendshape['weightMat'])) # N * 52

    return coeffs

def load_blendshape_obj(obj_path):
    '''
    Input : .obj file path.
    Output: V * 3 vertex
            F * 3 face
    '''

    with open(obj_path) as f:
        for line in f:
            els = line.split()
            if len(els) == 0:
                continue
            elif els[0] == 'v':
                float1, float2, float3 = float(els[1]), float(els[2]), float(els[3])
                try:
                    v_array = np.concatenate((v_array, np.array([[float1, float2, float3]])), axis=0)
                except:
                    v_array = np.array([[float1, float2, float3]])

            elif els[0] == 'f':
                try: # f v1 v2 v3
                    int1, int2, int3 = int(els[1]), int(els[2]), int(els[3])
                except: # f v1/v2 v2/v3 v3/v1
                    int1, int2, int3 = int(els[1].split('/')[0]), int(els[2].split('/')[0]), int(els[3].split('/')[0])
                try:
                    f_array = np.concatenate((f_array, np.array([[int1, int2, int3]])), axis=0)
                except:
                    f_array = np.array([[int1, int2, int3]])

    return v_array, f_array

def load_peoplesnapshot_hdf5(hdf5_path):
    '''
    Input : .hdf5 file path.
    '''
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        print(f)
