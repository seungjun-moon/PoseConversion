import os
import sys
import torch
import pickle
import argparse
import numpy as np
from utils.rotation_converter import batch_rodrigues, batch_euler2axis, batch_axis2euler, batch_matrix2axis, inverse_batch_rodrigues
from utils.load_params import load_smplx, load_flame
from pytorch3d.transforms import matrix_to_euler_angles

def add_smplx_flame(smplx_path, flame_path, return_axis=False):
    raise NotImplementedError