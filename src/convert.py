import os
import sys
import torch
import pickle
import argparse
import numpy as np
from utils.rotation_converter import batch_rodrigues, batch_euler2axis, batch_axis2euler, batch_matrix2axis, inverse_batch_rodrigues
from utils.load_params import load_smplx, load_flame
from pytorch3d.transforms import matrix_to_euler_angles

def smplx_to_smpl(smplx_path, return_axis=False):
    full_pose, cam, exp, shape = load_smplx(smplx_path)

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