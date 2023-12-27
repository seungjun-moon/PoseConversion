import os
import sys
import torch
import pickle
import argparse
import numpy as np
from pytorch3d.transforms import matrix_to_euler_angles

from src.load import load_pickle, load_integration
from src.convert import *
from src.refine import *
from utils.common import temporal_smooth, normalize_degree

def module_data_dict():
    data_dict = {}
    data_dict['gart']='smpl'
    data_dict['hood']='smpl'
    data_dict['hood2']='smplx'
    data_dict['scarf']='smplx'
    data_dict['next3d']='flame'
    data_dict['blendshape']='blendshape'

    return data_dict

def main(args):

    data_dict = module_data_dict()

    if args.action == 'convert':
        if args.load_source == 'smpl' and data_dict[args.module] == 'smplx':
            data = smpl_to_smplx(args.load_file)
        elif args.load_source == 'smpl' and data_dict[args.module] == 'flame':
            data = smpl_to_flame(args.load_file)
        elif args.load_source == 'smplx' and data_dict[args.module] == 'smpl':
            data = smplx_to_smpl(args.load_file)
        elif args.load_source == 'smplx' and data_dict[args.module] == 'flame':
            data = smplx_to_flame(args.load_file)
        elif args.load_source == 'flame' and data_dict[args.module] == 'smpl':
            data = flame_to_smpl(args.load_file)
        elif args.load_source == 'flame' and data_dict[args.module] == 'smplx':
            data = flame_to_smplx(args.load_file)
        elif args.load_source == 'blendshape' and data_dict[args.module] == 'flame':
            data = blendshape_to_flame(args.load_file)

        if args.smooth > 0:
            pose, cam, exp, shape = data
            _pose = temporal_smooth(pose, window=args.smooth)
            _exp  = temporal_smooth(exp,  window=args.smooth)
            data = _pose, cam, _exp, shape

        if args.module == 'scarf':
            smplx_for_SCARF(data, args.save_path)
        elif args.module == 'hood':
            smpl_for_HOOD(data, os.path.join(args.save_path, 'smpl_hood.pkl'))
        elif args.module == 'hood2':
            smplx_for_HOOD2(data, os.path.join(args.save_path, 'smplx_hood2.pkl'))
        elif args.module == 'next3d':
            flame_for_NEXT3D(data, os.path.join(args.save_path, 'flame_next3d.pkl'))
        elif args.module == 'gart':
            smpl_for_gart(data, os.path.join(args.save_path, 'smpl_gart.npy'))

    elif args.action == 'norm':
        # data = load_integration(args.load_file, datatype=args.load_source)
        # pose, _, _, _ = data
        # pose_ = normalize_degree(poses=pose[:, 3:], pi=1)
        # os.makedirs(args.save_path, exist_ok=True)
        # save_file = os.path.join(args.save_path, os.path.basename(args.load_file))
        # import shutil
        # shutil.copy(args.load_file, save_file)
        # data = dict(np.load(args.load_file))
        # data['body_pose'] = pose_
        
        # np.savez(save_file, **data)

        os.makedirs(args.save_path, exist_ok=True)
        save_file = os.path.join(args.save_path, os.path.basename(args.load_file))
        import shutil
        shutil.copy(args.load_file, save_file)

        data_ = dict(np.load(args.load_file))
        data = dict(np.load('/home/june1212/gart/data/insav_wild/male-3-wild/poses_optimized.npz'))

        data['betas'] = data_['betas']
        np.savez(save_file, **data)

    elif args.action == 'load':
        data = load_integration(args.load_file, datatype=args.load_source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_file', type=str, default='/home/june1212/PoseConversion/examples/smplx.pkl',
                        help='original pose path')
    parser.add_argument('--save_path', type=str, default='/home/june1212/PoseConversion/examples',
                        help='save path')
    parser.add_argument('--load_source', type=str, default='smpl',
                        help='loaded data source')
    parser.add_argument('--action', default='load',
                        help='')
    parser.add_argument('--module', type=str, default='hood', choices=['gart','hood', 'hood2', 'scarf','next3d'],
                        help='')
    parser.add_argument('--smooth', type=int, default=0,
                        help='designate smoothing window size. 0 for no smoothing. For now, only pose, exp smoothing applied.')
    args = parser.parse_args()
    main(args)
