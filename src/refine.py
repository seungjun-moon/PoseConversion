import os
import sys
import torch
import pickle
import argparse
import numpy as np
from utils.common import pickle_dump
from utils.rotation_converter import batch_rodrigues, batch_euler2axis, batch_axis2euler, batch_matrix2axis, inverse_batch_rodrigues
from utils.load_params import load_smplx, load_flame
from pytorch3d.transforms import matrix_to_euler_angles

# Refine to fit each module's output. Cannot load directly from .pkl dictionary. Utilize functions in load.py

def smplx_for_SCARF(smplx):
    raise NotImplementedError

def smpl_for_HOOD(smpl):
    assert len(smpl) == 4 # pose, cam, exp, shape

    out_dict = dict()

    ## KEY #1 : transl

    out_dict['transl'] = cam # TODO : is this validate?

    ## KEY #2 : body_pose

    rot_mats = pose[:,1:]
    rot_axis = torch.empty((rot_mats.shape[0],(rot_mats.shape[1]-1)*3)) # -1 for excluding global_rotation

    for i in range(len(rot_mats)):
        rot_axis[i] = matrix_to_euler_angles(rot_mats[i], convention="XYZ")

    out_dict['body_pose'] = rot_axis

    ## KEY #2 : global_orient

    rot_mats = pose[:,0]



