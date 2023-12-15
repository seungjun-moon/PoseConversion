import os
import cv2
import glob
import smplx
import logging
import numpy as np
from tqdm import tqdm
from os.path import join as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    BlendParams,
    SoftSilhouetteShader,
    SoftPhongShader,
    DirectionalLights,
    TexturesVertex
)
from pytorch3d.structures import Meshes

from utils.smpl_util import *
from src.load import load_pickle

DEVICE = "cuda"

def project(projection_matrices, keypoints_3d):
    p = torch.einsum("ij,mnj->mni", projection_matrices[:3, :3], keypoints_3d) + projection_matrices[:3, 3]
    p = p[..., :2] / p[..., 2:3]
    return p

def build_renderer(camera, IMG_SIZE):
    '''
    Currently support only intrinsic camera.
    Support for Extrinsic Cam from pickle will be updated.
    '''
    K = camera["intrinsic"]
    K = torch.from_numpy(K).float().to(DEVICE)

    R = torch.eye(3, device=DEVICE)[None]
    R[:, 0] *= -1
    R[:, 1] *= -1
    t = torch.zeros(1, 3, device=DEVICE)

    cameras = PerspectiveCameras(
        focal_length=K[None, [0, 1], [0, 1]],
        principal_point=K[None, [0, 1], [2, 2]],
        R=R,
        T=t,
        image_size=[IMG_SIZE],
        in_ndc=False,
        device=DEVICE,
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings = RasterizationSettings(
        image_size=IMG_SIZE, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size = 0, # to avoid the warning.
        )

    lights = DirectionalLights(ambient_color=((0.6, 0.6, 0.6),),direction=torch.Tensor([[0., -1., 0.]]), device=DEVICE)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=DEVICE,
            cameras=cameras,
            lights=lights
        )
    )
    return renderer


def draw_detect(frame: np.ndarray, poses2d: np.ndarray, color=(255, 255, 0)):
    for person in poses2d:
        # draw parts
        for (i, j) in BODY25_PART_PAIRS:
            if person.shape[-1] > 2 and min(person[[i, j], 2]) < 1e-3:
                continue
            frame = cv2.line(frame, tuple(person[i, :2].astype(int)),
                             tuple(person[j, :2].astype(int)), (color), 1)

        # draw joints
        for joint in person:
            if len(joint) > 2 and joint[-1] == 0:
                continue
            pos = joint[:2].astype(int)
            frame = cv2.circle(frame, tuple(pos), 2, (color), 2, cv2.FILLED)
    return frame

mesh_color_table = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
    'color0':[.7, .7, .6],
    'color2':[.5, .5, .7],
    'color1':[.7, .5, .5],
}

def set_mesh_color(verts_rgb, colors):
    if colors is None:
        colors = torch.Tensor(mesh_color_table['neutral'])
    if len(colors.shape) == 1:
        verts_rgb[:, :] = colors
    elif len(colors.shape) == 2:
        verts_rgb[:, :] = colors.unsqueeze(1)
    return verts_rgb

@torch.no_grad()
def main(args):

    root = args.load_path
    gender = args.gender
    keypoints_threshold = args.keypoints_threshold
    use_silhouette = args.silhouette
    downscale = args.downscale

    try:
        full_pose, cam, exp, shape = load_pickle(args.pickle_path)
    except:
        # logger.info('Load camera from {}'.format(camera_path))

    camera_path  = f"{root}/cameras.npz"
    smpl_path    = f"{root}/poses_optimized.npz"
    keypoints_2d = f"{root}/keypoints.npy"

    if os.path.isfile(camera_path):
        logger.info('Load camera from {}'.format(camera_path))
        camera = dict(np.load(camera_path))
    else:
        camera = 

    camera = dict(np.load(f"{root}/cameras.npz"))
    smpl_params = dict(np.load(f"{root}/poses_optimized.npz"))
    keypoints_2d = np.load(f"{root}/keypoints.npy")
    if downscale > 1:
        camera["intrinsic"][:2] /= downscale
    projection_matrices = camera["intrinsic"] @ camera["extrinsic"][:3]
    projection_matrices = torch.from_numpy(projection_matrices).float().to(DEVICE)

    # prepare data
    # joint_mapper = BODY25JointMapper()
    joint_mapper = SMPL2CUSTOMJointMapper()
    # smpl_params = dict(np.load(f"{root}/poses.npz"))
    
    keypoints_2d = torch.from_numpy(keypoints_2d).float().to(DEVICE)

    params = {}
    for k, v in smpl_params.items():
        if k == "thetas":
            tensor = torch.from_numpy(v[:, :3]).clone().to(DEVICE)
            params["global_orient"] = nn.Parameter(tensor)
            tensor = torch.from_numpy(v[:, 3:]).clone().to(DEVICE)
            params["body_pose"] = nn.Parameter(tensor)
        elif k == "betas":
            tensor = torch.from_numpy(v).clone().to(DEVICE)
            params[k] = nn.Parameter(tensor[None])
            # params[k] = tensor[None]
        else:
            tensor = torch.from_numpy(v).clone().to(DEVICE)
            params[k] = nn.Parameter(tensor)

    body_model = smplx.SMPL("./data/SMPLX/smpl", gender=gender)
    body_model.to(DEVICE)


    # masks = sorted(glob.glob(f"{root}/masks_sam/*"))
    # masks = [cv2.imread(p)[..., 0] for p in masks]
    masks = sorted(glob.glob(f"{root}/images/*"))
    masks = [cv2.imread(p) for p in masks]
    if downscale > 1:
        masks = [cv2.resize(m, dsize=None, fx=1/downscale, fy=1/downscale) for m in masks]
    masks = np.stack(masks, axis=0)

    img_size = masks[0].shape[:2]
    renderer = build_renderer(camera, img_size)

    os.makedirs(osp(root,'color_optimized'),exist_ok=True)
    os.makedirs(osp(root,'color'),exist_ok=True)
    os.makedirs(osp(root,'mesh_aligned_optimized'),exist_ok=True)
    os.makedirs(osp(root,'mesh_aligned'),exist_ok=True)
    for i in range(len(masks)):
        # mask = torch.from_numpy(masks[i:i+1]).float().to(DEVICE) / 255
        ori_img = masks[i] # (708, 1259, 3)

        # silhouette loss
        smpl_output = body_model(
            betas=params["betas"].clone().detach(),
            global_orient=params["global_orient"][i:i+1],
            body_pose=params["body_pose"][i:i+1],
            transl=params["transl"][i:i+1],
        )

        # keypoints loss
        keypoints_pred = project(projection_matrices, joint_mapper(smpl_output))

        verts_rgb = torch.ones_like(smpl_output.vertices)
        # colors = torch.Tensor(mesh_color_table['neutral'])
        colors = torch.Tensor(mesh_color_table['pink'])
        verts_rgb = set_mesh_color(verts_rgb, colors)
        textures = TexturesVertex(verts_features=verts_rgb)
        meshes = Meshes(
            verts=smpl_output.vertices,
            faces=body_model.faces_tensor[None].repeat(1, 1, 1),
            textures=textures
        )

        img = renderer(meshes)
        # silhouette = renderer(meshes)[..., 3]
        # color = renderer(meshes)[..., :3]
        color_img = img[0,...,:3].cpu().detach().numpy()
        color_img *= 255
        # color_path = osp(root,'color',f'color_{i:04d}.png')
        # cv2.imwrite(color_path,color_img)

        thresh = 0.0
        mask_img = img[0,...,3].cpu().detach().numpy()
        mask_img = mask_img > thresh
        mask_img = mask_img[...,None]

        vis_img = ori_img*~mask_img + color_img*mask_img
        cv2.imwrite(args.save_path, vis_img)
        # print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=False)
    parser.add_argument("--save_path", type=str, required=False)

    parser.add_argument("--pickle_path", type=str, required=False)

    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--keypoints-threshold", type=float, default=0.2)
    parser.add_argument("--silhouette", action="store_true")
    parser.add_argument("--downscale", type=float, default=1.0)
    args = parser.parse_args()

    main(args)
