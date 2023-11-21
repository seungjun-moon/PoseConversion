import os
import torch
import smplx
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from src.load import load_pickle, load_obj
from utils.util import visualize_grid
from utils.common import batch_matrix2euler

def main(args,
         ext='npz',
         gender='neutral',
         plot_joints=False,
         sample_expression=True,
         num_expression_coeffs=10,
         use_face_contour=False):

    os.makedirs(args.save_path, exist_ok=True)

    if args.model_type != 'mesh':
        full_pose, _, exp, shape = load_pickle(args.load_path)
        full_pose = batch_matrix2euler(full_pose).reshape(full_pose.shape[0], -1)
        num_frames = full_pose.shape[0]

    else: #mesh
        mesh_list = []
        for file in os.listdir(args.load_path):
            if file.endswith('.obj'):
                mesh_list.append(file)
        num_frames = len(mesh_list)

    if args.model_type == 'smplx':
        global_orient = full_pose[:,:3]
        body_pose = full_pose[:,3:]
        part_body_pose = body_pose[:,:21*3]
        jaw_pose = body_pose[:,21*3:22*3]
        leye_pose = body_pose[:,22*3:23*3]
        reye_pose = body_pose[:,23*3:24*3]
        left_hand_pose = body_pose[:,24*3:39*3]
        right_hand_pose = body_pose[:,39*3:54*3]

    if args.model_type in ['smplx', 'smpl']:

        num_betas = shape.shape[-1]

        model = smplx.create(args.model_folder, model_type=args.model_type,
                             gender=gender, use_face_contour=use_face_contour,
                             num_betas=num_betas,
                             num_expression_coeffs=num_expression_coeffs,
                             ext=ext)

    for i in tqdm(range(num_frames)):
        if args.model_type == 'smpl':
            output = model(betas=shape.unsqueeze(0), expression=exp[i].unsqueeze(0),
                           return_verts=True)
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            # joints = output.joints.detach().cpu().numpy().squeeze()
        elif args.model_type == 'smplx':
            print(global_orient.shape[i:i+1])
            print(shape.unsqueeze(0).shape)
            output = model(betas=shape.unsqueeze(0), expression=exp[i].unsqueeze(0), global_orient=global_orient[i:i+1], \
                           body_pose=part_body_pose[i:i+1], jaw_pose=jaw_pose[i:i+1], leye_pose=leye_pose[i:i+1], \
                           reye_pose=reye_pose[i:i+1], left_hand_pose=left_hand_pose[i:i+1], right_hand_pose=right_hand_pose[i:i+1])
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            # joints = output.joints.detach().cpu().numpy().squeeze()
        elif args.model_type == 'flame':
            raise NotImplementedError
        elif args.model_type == 'mesh':
            vertices, faces = load_obj(os.path.join(args.load_path, mesh_list[i]))

        if args.plotting_module == 'pyrender':
            import pyrender
            import trimesh
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
            tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                       vertex_colors=vertex_colors)

            mesh = pyrender.Mesh.from_trimesh(tri_mesh)

            scene = pyrender.Scene()
            scene.add(mesh)

            if plot_joints:
                sm = trimesh.creation.uv_sphere(radius=0.005)
                sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
                tfs = np.tile(np.eye(4), (len(joints), 1, 1))
                tfs[:, :3, 3] = joints
                joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(joints_pcl)

            pyrender.Viewer(scene, use_raymond_lighting=True)

        elif args.plotting_module == 'matplotlib':
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
            face_color = (1.0, 1.0, 0.9)
            edge_color = (0, 0, 0)
            mesh.set_edgecolor(edge_color)
            mesh.set_facecolor(face_color)
            ax.add_collection3d(mesh)
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

            if plot_joints:
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
            plt.show()

        elif args.plotting_module == 'open3d':
            import open3d as o3d

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(
                vertices)
            mesh.triangles = o3d.utility.Vector3iVector(model.faces)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.3, 0.3, 0.3])

            geometry = [mesh]
            if plot_joints:
                joints_pcl = o3d.geometry.PointCloud()
                joints_pcl.points = o3d.utility.Vector3dVector(joints)
                joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
                geometry.append(joints_pcl)

            o3d.visualization.draw_geometries(geometry)

        elif args.plotting_module == 'pytorch3d':

            device = 'cuda'

            import cv2
            import pytorch3d
            from pytorch3d.structures import Meshes, join_meshes_as_scene
            from pytorch3d.renderer import (
                look_at_view_transform,
                FoVPerspectiveCameras, 
                PointLights, 
                DirectionalLights, 
                Materials, 
                RasterizationSettings, 
                MeshRenderer, 
                MeshRasterizer,  
                SoftPhongShader,
                TexturesUV,
                TexturesVertex
            )
            from pytorch3d.renderer.mesh.textures import Textures

            if args.model_type in ['smplx', 'smpl']:
                R, T = look_at_view_transform(3, 0, 0)
            else:
                R, T = look_at_view_transform(0.7, 0, 0)

            T[0,0] = T[0,0]
            T[0,1] = T[0,1]
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

            raster_settings = RasterizationSettings(
                image_size=1024, 
                blur_radius=0.0, 
                faces_per_pixel=1,
            )
            lights = PointLights(device=device, location=[[0.0, 0.0, +3.0]])
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device, 
                    cameras=cameras,
                    lights=lights
                )
            )
            
            if args.model_type in ['smplx', 'smpl']:
                faces = model.faces.astype(np.int64)

            vertices = torch.from_numpy(vertices).unsqueeze(0).to(device).float()
            faces = torch.from_numpy(faces).unsqueeze(0).to(device)
            verts_rgb = torch.full(vertices.shape, 0.5)
            textures = Textures(verts_rgb=verts_rgb.to(device))
            
            mesh = Meshes(verts = vertices,
                          faces = faces,
                          textures=textures)

            image = renderer(mesh)
            image = 255*image[0, ..., :3].cpu().numpy()
            cv2.imwrite(os.path.join(args.save_path, '{}.png'.format(str(i).zfill(4))), image)

        else:
            raise ValueError('Unknown plotting_module: {}'.format(args.plotting_module))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL Families Parameter Visualization')

    parser.add_argument('--model_folder', type=str,
                        help='The path to the model folder')
    parser.add_argument('--model_type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame', 'mesh'],
                        help='The type of model to load')
    parser.add_argument('--load_path', default='./examples/smplx.pkl', type=str,
                        help='The path for the target pose sequence dictionary')
    parser.add_argument('--save_path', default='./results', type=str,
                        help='Path to save render results')
    parser.add_argument('--plotting_module', default='pytorch3d', type=str,
                        choices=['pyrender', 'matplotlib', 'open3d', 'pytorch3d'],
                        help='Tool for the visualization')

    args = parser.parse_args()
    main(args)
    
