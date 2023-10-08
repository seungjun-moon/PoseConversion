import os.path as osp
import argparse

import numpy as np
import torch

import smplx


def main(args,
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):

    model = smplx.create(args.model_folder, model_type=args.model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
   	print(model)
    