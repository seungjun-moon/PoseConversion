import os
import sys
import torch
import pickle
import argparse
import numpy as np
from utils.common import pickle_dump, batch_euler2matrix, batch_matrix2euler, batch_matrix2axis

# Refine to fit each module's output. Cannot load directly from .pkl dictionary. Utilize functions in load.py

def smplx_for_SCARF(smplx, save_path):
    raise NotImplementedError

def smpl_for_HOOD(smpl, save_path):
    assert len(smpl) == 4 # pose, cam, exp, shape

    pose, cam, exp, shape = smpl

    out_dict = dict()

    ## KEY #1 : transl

    out_dict['transl'] = cam

    ## KEY #2 : body_pose

    rot_mats = pose[:, 1:]

    rot_axis = batch_matrix2axis(rot_mats)


    ### TODO

    # print(rot_mats[0,0])
    # from smplx.lbs import batch_rodrigues
    # print(rot_axis.shape)
    # new_rot_mats = batch_rodrigues(rot_axis[0])
    # print(new_rot_mats[0])

    rot_axis = rot_axis.reshape(rot_axis.shape[0], -1)

    # rot_axis[:,18*3:19*3]=0 #TODO : joint #18 becomes zero. --> related to right elbow.
    # rot_axis[:,19*3:20*3]=0 #TODO : joint #19 becomes zero.

    out_dict['body_pose'] = rot_axis

    ## KEY #3 : global_orient

    grot_mats = pose[:,0]
    grot_axis = batch_matrix2euler(grot_mats)

    ## TODO: Alignment between PIXIE and HOOD are different.
    grot_axis = torch.zeros(grot_axis.shape)


    out_dict['global_orient'] = grot_axis

    ## KEY #4 : betas

    out_dict['betas'] = shape[:10]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # for key in out_dict.keys():
    #     print(out_dict[key].shape)

    print('Save SMPL for HOOD in {}'.format(save_path))
    pickle_dump(out_dict, save_path)

    return out_dict




