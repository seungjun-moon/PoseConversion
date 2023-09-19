import os
import sys
import torch
import pickle
import argparse
import numpy as np
from pytorch3d.transforms import matrix_to_euler_angles

from src.load import load_smplx, load_smpl, load_flame
from src.convert import *
from src.refine import *


def module_data_dict():
    data_dict = {}
    data_dict['hood']='smpl'
    data_dict['scarf']='smplx'
    data_dict['flame']='flame'

    return data_dict

def main(args):

    data_dict = module_data_dict()

    if args.load_source == 'smpl' and data_dict[args.module] == 'smpl':
        data = load_smpl(args.load_path)
    elif args.load_source == 'smpl' and data_dict[args.module] == 'smplx':
        data = smpl_to_smplx(args.load_path)
    elif args.load_source == 'smpl' and data_dict[args.module] == 'flame':
        data = smpl_to_flame(args.load_path)
    elif args.load_source == 'smplx' and data_dict[args.module] == 'smpl':
        data = smplx_to_smpl(args.load_path)
    elif args.load_source == 'smplx' and data_dict[args.module] == 'smplx':
        data = load_smplx(args.load_path)
    elif args.load_source == 'smplx' and data_dict[args.module] == 'flame':
        data = smplx_to_flame(args.load_path)
    elif args.load_source == 'flame' and data_dict[args.module] == 'smpl':
        data = flame_to_smpl(args.load_path)
    elif args.load_source == 'flame' and data_dict[args.module] == 'smplx':
        data = flame_to_smplx(args.load_path)
    elif args.load_source == 'flame' and data_dict[args.module] == 'flame':
        data = load_flame(args.load_path)

    if args.module == 'scarf':
        smplx_for_SCARF(data, args.save_path)
    elif args.module == 'hood':
        smpl_for_HOOD(data, os.path.join(args.save_path, 'smpl_hood.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='/home/june1212/PoseConversion/examples/smplx.pkl', help='original pose path')
    parser.add_argument('--save_path', type=str, default='/home/june1212/scarflame/examples', help='save path')
    parser.add_argument('--load_source', type=str, default='smplx', choices=['smpl','smplx','flame'], help='loaded data source')
    parser.add_argument('--module', type=str, default='hood', choices=['hood','scarf','next3d'], help='data usage')
    args = parser.parse_args()
    main(args)
