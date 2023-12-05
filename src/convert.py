import os
import sys
import torch
import pickle
import argparse
import numpy as np
from utils.rotation_converter import batch_rodrigues, batch_euler2axis, batch_axis2euler, batch_matrix2axis, inverse_batch_rodrigues
from src.load import load_pickle, load_json

def smplx_to_smpl(smplx_path, save=False):
    smplx_pose, cam, exp, shape = load_pickle(smplx_path)

    n_frames = len(smplx_pose)
    smpl_indices_in_smplx = [i for i in range(22)]+[25,40]

    smpl_pose = torch.empty((n_frames, 24, 3, 3))

    for frame in range(n_frames):
        smpl_pose[frame] = smplx_pose[frame,smpl_indices_in_smplx] #55*3*3 --> 24*3*3

    if save:
        raise NotImplementedError

    return smpl_pose, cam, exp, shape

def smplx_to_flame(smplx_path, return_axis=False):
    raise NotImplementedError

def flame_to_smplx(flame_path, return_axis=False):
    raise NotImplementedError

def flame_to_smpl(flame_path, return_axis=False):
    raise NotImplementedError

def smpl_to_smplx(smpl_path, return_axis=False):
    raise NotImplementedError

def smpl_to_flame(smpl_path, return_axis=False):
    raise NotImplementedError

def blendshape_to_flame(blendshape_path):
    '''
    flame_pose : N * joints * 3
    flame_shape : 100
    flame_exp : N * 50
    flame_cam : N * 3 (for the orthographic)
    '''

    coeffs = load_json(blendshape_path)
    num = coeffs.shape[0]

    flame_pose  = torch.zeros(num, 3*4)
    shape = torch.zeros(num, 100)
    exp   = torch.zeros(num,  50)
    cam   = torch.zeros(num,   3)

    # mouth open

    flame_pose[:,3] = coeffs[:,17]/2 + coeffs[:,19]/2
    flame_pose[:,3] = torch.clamp(flame_pose[:,3], max=0.2)

    # mouth funnel

    exp[:,0] = torch.clamp((coeffs[:,14]+coeffs[:,19]-coeffs[:,33]/5+coeffs[:,34]/5) * -3.0 -0.4, max=2.0, min=-2.0)
    exp[:,1] = torch.clamp((coeffs[:,14]+coeffs[:,19]-coeffs[:,33]/5+coeffs[:,34]/5) * +1.5 +0.4, max=0.8, min=-0.8)

    # eye blink, suppose 60 FPS --> 150 Frame

    v=10
    for i in range(v, num):
        weight = max(150-i%150-(150-v), i%150-(150-v), 0) * 2 # make periodic peak

        print(weight)
        
        exp[i,2]  = weight *  0.03
        exp[i,3]  = weight *  0.12
        exp[i,8]  = weight *  0.08
        exp[i,9]  = weight *  0.15
        exp[i,11] = weight *  0.07
        exp[i,14] = weight *  0.10
        exp[i,19] = weight *  0.14
        exp[i,22] = weight *  0.10
        exp[i,25] = weight *  0.05
        exp[i,30] = weight *  0.05
        exp[i,34] = weight *  0.04
        exp[i,36] = weight *  0.12
        exp[i,41] = weight *  0.04
        exp[i,42] = weight *  0.04
        exp[i,45] = weight *  0.04
        exp[i,46] = weight *  0.04
        exp[i,48] = weight *  0.03


        exp[i,4]  = weight * -0.12
        exp[i,5]  = weight * -0.12
        exp[i,7]  = weight * -0.12
        exp[i,15] = weight * -0.07
        exp[i,16] = weight * -0.14
        exp[i,17] = weight * -0.03
        exp[i,18] = weight * -0.09
        exp[i,21] = weight * -0.04
        exp[i,24] = weight * -0.05
        exp[i,33] = weight * -0.07
        exp[i,40] = weight * -0.06
        exp[i,47] = weight * -0.01

    return flame_pose, cam, exp, shape





