import os
import sys
import torch
import pickle
import argparse
import numpy as np
from utils.rotation_converter import batch_rodrigues, batch_euler2axis, batch_axis2euler, batch_matrix2axis, inverse_batch_rodrigues
from src.load import load_pickle

def smplx_to_smpl(smplx_path, save=False):
    smplx_pose, cam, exp, shape = load_pickle(smplx_path)

    n_frames = len(smplx_pose)
    smpl_indices_in_smplx = [i for i in range(22)]+[25,40]

    smpl_pose = torch.empty((n_frames, 24, 3, 3))

    for frame in range(n_frames):
        smpl_pose[frame] = smplx_pose[frame,smpl_indices_in_smplx] # 55*3*3 --> 24*3*3

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