import os
import numpy as np
import torch
import pickle
import argparse

from utils.common import batch_euler2matrix, save_pkl

def flame_from_next3d(deca_path):
    '''
    Integrate FLAME parameters extracted by DECA. This follows the format of saving format of DECA in Next3D.
    '''

    frames = sorted(os.listdir(deca_path))

    full_pose = torch.zeros((len(frames),4,3))
    cam       = torch.zeros((len(frames), 3))
    shape     = torch.zeros((len(frames), 100))
    exp       = torch.zeros((len(frames), 50))

    for i,frame in enumerate(frames):
        flame = np.load(os.path.join(deca_path, frame, frame+'.npy'), allow_pickle=True)
        posecode  = flame.item().get('pose') # 1*6
        camcode   = flame.item().get('cam') # 1*3
        shapecode = flame.item().get('shape') # 1*100
        expcode   = flame.item().get('exp') # 1*50
        eye_posecode = flame.item().get('eye_pose') # 1*6

        euler_jaw_pose = torch.from_numpy(posecode[:,3:])
        euler_neck_pose = torch.from_numpy(posecode[:,:3])

        euler_eye_pose_1 = torch.from_numpy(eye_posecode[:,:3])
        euler_eye_pose_2 = torch.from_numpy(eye_posecode[:,3:])

        full_pose[i] = torch.cat((euler_jaw_pose, euler_neck_pose, euler_eye_pose_1, euler_eye_pose_2), dim=0)
        cam[i] = torch.from_numpy(camcode)
        shape[i] = torch.from_numpy(shapecode)
        exp[i] = torch.from_numpy(expcode)

    full_pose = batch_euler2matrix(full_pose)

    flame_dict = {}
    flame_dict['full_pose'] = full_pose.numpy() # N * 4 * 3 * 3
    flame_dict['cam'] = cam.numpy() # N * 3
    flame_dict['shape'] = shape.numpy() # N * 100
    flame_dict['exp'] = exp.numpy() # N * 50

    return flame_dict

def main(args):
    flame_dict = flame_from_next3d(args.raw_path)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    save_pkl(os.path.join(args.save_path, 'flame.pkl'), flame_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='/home/june1212/next3d/dataset_preprocessing/ffhq/video1/deca_results/', help='raw data path')
    parser.add_argument('--save_path', type=str, default='/home/june1212/PoseConversion/examples/flame/', help='save path')
    args = parser.parse_args()
    main(args)

