import os
import torch
import smplx
import pickle
import argparse
import numpy as np

from src.load import load_pickle
from utils.util import visualize_grid

def main(args,
         ext='npz',
         gender='neutral',
         plot_joints=False,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):

    full_pose, _, exp, shape = load_pickle(args.load_path)
    num_betas = shape.shape[-1]

    model = smplx.create(args.model_folder, model_type=args.model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    os.makedirs(args.save_path, exist_ok=True)

    for i in range(full_pose.shape[0]):
        output = model(betas=shape.unsqueeze(0), expression=exp[i].unsqueeze(0),
                       return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        if i==0:
            print('Vertices shape =', vertices.shape)
            print('Joints shape =', joints.shape)

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
            import pytorch3d
            from src.rendering import Pytorch3dRasterzier

            renderer = Pytorch3dRasterzier(device='cuda')
            vis_dict={
            'image' : renderer(vertices)
            }

            visualize_grid(vis_dict, os.path.join(args.save_path, '{}.png'.format(str(i).zfill(4))))

            ### fill in here ###




        else:
            raise ValueError('Unknown plotting_module: {}'.format(plotting_module))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL Families Parameter Visualization')

    parser.add_argument('--model_folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model_type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
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
    
