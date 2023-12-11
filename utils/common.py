import os
import numpy as np
import torch
import pickle
from utils.rotation_converter import *
from pytorch3d.transforms import matrix_to_euler_angles, matrix_to_axis_angle

def batch_euler2matrix(pose):
    '''
    Input : Pose Tensor [N * J * 3]

    Output : Pose Tensor [N * J * 3 * 3]
    '''

    out = torch.empty((pose.shape[0], pose.shape[1], pose.shape[2], pose.shape[2]))

    for i in range(len(pose)): #N
        out[i] = batch_rodrigues(batch_euler2axis(pose[i]))

    return out

def batch_matrix2euler(pose):
    '''
    Input : Pose Tensor [N * J * 3 * 3]

    Output : Pose Tensor [N * J * 3]
    '''
    return matrix_to_euler_angles(pose, convention="XYZ")

def batch_matrix2axis(pose):
    '''
    Input : Pose Tensor [N * J * 3 * 3]

    Output : Pose Tensor [N * J * 3]
    '''
    return matrix_to_axis_angle(pose)

def pickle_dump(loadout, file):
    '''
    Dump a pickle file. Create the directory if it does not exist.
    '''
    os.makedirs(os.path.dirname(str(file)), exist_ok=True)

    with open(file, 'wb') as f:
        pickle.dump(loadout, f)

def save_pkl(savepath, params, ind=0):
    out_data = {}
    for k, v in params.items():
        if torch.is_tensor(v):
            out_data[k] = v[ind].detach().cpu().numpy()
        else:
            out_data[k] = v
    # import ipdb; ipdb.set_trace()
    with open(savepath, 'wb') as f:
        pickle.dump(out_data, f, protocol=2)


def fit_pose_length(element, length):
    '''
    Fit element length without discontinuity, to the number of length. 
    '''

    element_concat = element
    element_flip = torch.flip(element, dims=[0])

    mul = length // len(element)

    for i in range(1,mul+1):
        if i % 2 == 1:
            element_concat = torch.cat((element_concat, element_flip[1:]), dim=0)
        else:
            element_concat = torch.cat((element_concat, element[1:]), dim=0)

    return element_concat[:length]

def temporal_smooth(param, window=3):
    '''
    param: N * X1 * X2 or N * X1
    _param: param.shape
    '''
    _param = param.clone().detach()

    weights = torch.FloatTensor([1 - float(abs(i-window))/(window+1) for i in range(2*window+1)]).to(_param.device)

    assert param.shape[0] > 2 * window, 'Frame number is too small for smoothing'

    for i in range(window,param.shape[0]-window):
        _param[i] = torch.matmul(weights,param[i-window:i+window+1])/torch.sum(weights)

    return _param

def find_affine_transform(P1, P2):
    '''
    2D affine transformation
    P1: numpy array with N * 2
    P2: numpy array with N * 2

    return: Affine Matrix
    '''

    A = np.vstack((P1.T, np.ones((1, P1.shape[0])))).T
    B = np.vstack((P2.T, np.ones((1, P2.shape[0])))).T

    # Solve for the transformation matrix
    T, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Reshape the matrix to a 3x3 matrix
    # transform_matrix = np.vstack((T, [0, 0, 1]))
    
    transform_matrix = T

    return transform_matrix


