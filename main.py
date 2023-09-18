import os
import sys
import torch
import pickle
import argparse
import numpy as np
from utils.rotation_converter import batch_rodrigues, batch_euler2axis, batch_axis2euler, batch_matrix2axis, inverse_batch_rodrigues
from src.load import load_smplx, load_flame
from pytorch3d.transforms import matrix_to_euler_angles
from src.convert import *

def main(args):

    a,b,c,d = smplx_to_smpl('examples/smplx.pkl')

    # full_pose, exp, cam = load_smplx(args.smpl_path)

    # pose1 = full_pose[150,0:1]

    # print(pose1)
    # print('')


    # inv = matrix_to_euler_angles(pose1, convention="XYZ")
    # inv = batch_euler2axis(inv)
    # inv = batch_rodrigues(inv)
    # print(inv)
    # print('')

def flame(args):

    full_pose, exp, cam = load_pixie_smplx(args.smpl_path)

    print(full_pose.shape)
    print(exp.shape)
    print(cam.shape)


    count = -1
    for dirname in sorted(os.listdir(args.deca_path)):
        if 'mirror' in dirname:
            continue
        try:
            flame = np.load(os.path.join(args.deca_path, dirname, dirname+'.npy'), allow_pickle=True)
            count +=1
        except:
            print('File Not Found : ',os.path.join(args.deca_path, dirname, dirname+'.npy'))
            continue

        posecode = flame.item().get('pose') # 1*3
        euler_jaw_pose = torch.from_numpy(posecode[:,3:])
        axis_jaw_pose  = batch_euler2axis(euler_jaw_pose)
        rot_jaw_pose   = batch_rodrigues(axis_jaw_pose)

        axis_neck_pose = torch.from_numpy(posecode[:,:3])
        rot_neck_pose  = batch_rodrigues(axis_neck_pose)

        axis_id_pose = torch.zeros(axis_neck_pose.shape)
        rot_id_pose  = batch_rodrigues(axis_id_pose)

        try:
            full_pose[count,0] = torch.FloatTensor([[ 1.0, -0.0,  0.0],
                                                    [-0.0, -1.0,  0.0],
                                                    [ 0.0, -0.0, -1.0]])

            # for i in range(1,10): # --> no head movement
            #     full_pose[count,(i+5)+1] = rot_neck_pose[0] # 12 for neck pose

            full_pose[count,3] = rot_id_pose[0]
            full_pose[count,6] = rot_id_pose[0]
            full_pose[count,9] = rot_id_pose[0]

            full_pose[count,12] = rot_id_pose[0] # 12 for neck pose, make it identity
            full_pose[count,15] = rot_neck_pose[0] # 15 for head pose. PUT NECK POSE in here!
            full_pose[count,22]   = rot_jaw_pose[0]  # 22 for jaw pose

        except:
            print('{} frames are saved'.format(count))
            break

    print(full_pose.shape)

    pose_dict = {}
    pose_dict['full_pose'] = full_pose.numpy()
    pose_dict['cam'] = cam
    pose_dict['exp'] = exp

    sys.path.insert(0, '/home/june1212/scarflame/process_data/submodules/PIXIE')
    from pixielib.utils import util
    util.save_pkl(args.smpl_path[:-4]+'+flame.pkl', pose_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deca_path', type=str, default='/home/june1212/next3d/dataset_preprocessing/ffhq/video5/deca_results', help='deca result path')
    parser.add_argument('--smpl_path', type=str, default='/home/june1212/scarflame/data/pixie_radioactive.pkl', help='original pose path')
    args = parser.parse_args()
    main(args)
