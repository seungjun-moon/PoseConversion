import os
import numpy as np
import torch
import pickle
import argparse

from utils.common import batch_euler2matrix, save_pkl

def flame_from_next3d(deca_path, save_rot=True):
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
        

        euler_neck_pose = torch.from_numpy(posecode[:,:3])
        euler_jaw_pose  = torch.from_numpy(posecode[:,3:])

        try:
            eye_posecode = flame.item().get('eye_pose') # 1*6
            euler_eye_pose_1 = torch.from_numpy(eye_posecode[:,:3])
            euler_eye_pose_2 = torch.from_numpy(eye_posecode[:,3:])
        except:
            print('No eye pose provided!')
            euler_eye_pose_1 = torch.zeros(posecode[:,:3].shape)
            euler_eye_pose_2 = torch.zeros(posecode[:,:3].shape)

        full_pose[i] = torch.cat((euler_neck_pose, euler_jaw_pose, euler_eye_pose_1, euler_eye_pose_2), dim=0)
        cam[i] = torch.from_numpy(camcode)
        shape[i] = torch.from_numpy(shapecode)
        exp[i] = torch.from_numpy(expcode)


    if save_rot:
        full_pose = batch_euler2matrix(full_pose)
    else:
        full_pose = full_pose.reshape((full_pose.shape[0], -1))

    return full_pose, cam, exp, shape

def smpl_from_gart(smpl_path, save_rot=False):
    '''
    '''
    full_pose = torch.from_numpy(np.load(smpl_path)) # N * 72
    cam   = torch.zeros(full_pose.shape[0], 3)
    exp   = torch.zeros(full_pose.shape[0], 50)
    shape = torch.zeros(full_pose.shape[0], 100)

    if save_rot:
        full_pose = batch_euler2matrix(full_pose)

    return full_pose, cam, exp, shape

def main(args):

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    pkl_dict = {}

    if module_name == 'next3d':
        full_pose, cam, exp, shape = flame_from_next3d(args.data_path, save_rot=False)

    elif module_name == 'gart':
        full_pose, cam, exp, shape = smpl_from_gart(args.data_path, save_rot=False)

    pkl_dict = {}
    pkl_dict['full_pose'] = full_pose.numpy() # N * 4 * 3 * 3 if save_rot, else N * 4 * 3
    pkl_dict['cam'] = cam.numpy() # N * 3
    pkl_dict['exp'] = exp.numpy() # N * 50
    pkl_dict['shape'] = shape.numpy() # N * 100

    save_pkl(args.save_path, pkl_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module_name', type=str, default='gart', help='')
    parser.add_argument('--data_path', type=str, default='/home/june1212/gart/novel_poses/walking.npy', help='')
    parser.add_argument('--save_path', type=str, default='/home/june1212/gart/novel_poses/pickle.pkl', help='')
    args = parser.parse_args()
    main(args)

