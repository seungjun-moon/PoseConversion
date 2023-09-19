import os
import sys
import torch
import pickle
import argparse
import numpy as np
from utils.common import pickle_dump, batch_euler2matrix, batch_matrix2euler

# Refine to fit each module's output. Cannot load directly from .pkl dictionary. Utilize functions in load.py

def smplx_for_SCARF(smplx, save_path):
    raise NotImplementedError

def smpl_for_HOOD(smpl, save_path):
    assert len(smpl) == 4 # pose, cam, exp, shape

    pose, cam, exp, shape = smpl

    out_dict = dict()

    ## KEY #1 : transl

    out_dict['transl'] = cam # TODO : is this validate?

    ## KEY #2 : body_pose

    rot_mats = pose[:, 1:]
    rot_axis = batch_matrix2euler(rot_mats)

    rot_axis = rot_axis.reshape(rot_axis.shape[0], -1)

    out_dict['body_pose'] = rot_axis

    ## KEY #3 : global_orient

    grot_mats = pose[:,0]
    grot_axis = batch_matrix2euler(grot_mats)

    out_dict['global_orient'] = grot_axis

    ## KEY #4 : betas

    out_dict['betas'] = shape[:10]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # for key in out_dict.keys():
    #     print(out_dict[key].shape)

    print('Save SMPL for HOOD in {}'.format(save_path))
    pickle_dump(out_dict, save_path)

    return out_dict




