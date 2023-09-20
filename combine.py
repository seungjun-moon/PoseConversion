import os
import sys
import torch
import pickle
import argparse
import numpy as np
from src.load import load_pickle
from utils.common import fit_pose_length, save_pkl, batch_euler2matrix


def add_smpl_flame(smpl, flame, pivot='face'):
    raise NotImplementedError

def add_smplx_flame(smplx, flame, pivot='face'):
    '''
    Input : Tuples with (full_pose, cam, exp, shape)

    Output : full_pose, cam, exp, shape
    '''

    pose1, cam1, exp1, shape1 = smplx
    pose2, cam2, exp2, shape2 = flame
    
    length = len(pose2) if pivot == 'face' else len(pose1)

    pose1 = fit_pose_length(pose1, length)
    pose2 = fit_pose_length(pose2, length)

    cam1 = fit_pose_length(cam1, length)
    cam2 = fit_pose_length(cam2, length)

    exp1 = fit_pose_length(exp1, length)
    exp2 = fit_pose_length(exp2, length)

    pose1[:,15] = pose2[:,0] # FLAME neck_pose --> SMPL-X head pose
    pose1[:,22] = pose2[:,1] # FLAME jaw_pose  --> SMPL-X jaw pose

    # TODO : As of now, fix to the identity

    gpose_matrix = torch.FloatTensor([[ 1.0, -0.0,  0.0],
                                   [-0.0, -1.0,  0.0],
                                   [ 0.0, -0.0, -1.0]])

    gpose_matrices = gpose_matrix.repeat(length,1,1)

    pose1[:,0] = gpose_matrices

    id_poses = batch_euler2matrix(torch.zeros((length,1,3))).squeeze(1)

    pose1[:,3] = id_poses
    pose1[:,6] = id_poses
    pose1[:,9] = id_poses
    pose1[:,12] = id_poses

    return pose1, cam1, exp1, shape1

def main(args):
    ## TODO : Generalized logic
    source1 = load_pickle(args.smplx_path)
    source2 = load_pickle(args.flame_path)

    full_pose, cam, exp, shape = add_smplx_flame(source1, source2)

    combine_dict = {}
    combine_dict['full_pose'] = full_pose.numpy() # N * 4 * 3 * 3
    combine_dict['cam'] = cam.numpy() # N * 3
    combine_dict['exp'] = exp.numpy() # N * 50
    combine_dict['shape'] = shape.numpy() # N * 100

    save_pkl(os.path.join(args.save_path, 'smplx+flame.pkl'), combine_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_path',  type=str, default=None, help='')
    parser.add_argument('--smplx_path', type=str, default='/home/june1212/PoseConversion/examples/smplx.pkl', help='')
    parser.add_argument('--flame_path', type=str, default='/home/june1212/PoseConversion/examples/flame/flame.pkl', help='')

    parser.add_argument('--save_path', type=str, default='/home/june1212/PoseConversion/examples', help='save path')
    args = parser.parse_args()
    main(args)


