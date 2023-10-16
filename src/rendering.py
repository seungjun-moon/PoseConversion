import os
import torch
import torch.nn as nn
import pytorch3d

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
# from pytorch3d.renderer import RasterizationSettings

from utils.util import face_vertices

class Pytorch3dRasterzier(nn.Module):
    def __init__(self, device='cuda:0'):
        super(Pytorch3dRasterzier, self).__init__()
        self.image_size = 512
        self.device=device
        self.batch_size = 1
        self.setup_render()

    def setup_render(self, R=torch.eye(3), T=torch.zeros([1,3])):
        R = R.unsqueeze(0)
        self.cameras = pytorch3d.renderer.cameras.FoVOrthographicCameras(
                       R=R.expand(self.batch_size, -1, -1), T=T.expand(self.batch_size, -1), znear=0.0).to(self.device)
        _, faces, _ = load_obj('/home/june1212/scarf/data/SMPL_X_template_FLAME_uv.obj')
        self.faces = faces.verts_idx[None,...].to(self.device)

        faces = self.faces.expand(self.batch_size,-1,-1)

        print(faces.shape)

        tex = torch.ones((1, self.faces.shape[1], 3)).to(self.device)

        self.attributes=face_vertices(tex, faces)

    def forward(self, vertices):
        vertices = torch.from_numpy(vertices).unsqueeze(0).to(self.device)
        image = pytorch3d_rasterize(vertices = vertices, faces = self.faces.expand(self.batch_size, -1, -1),
                                    image_size = self.image_size, attributes = self.attributes)

        print(image[:,:3].shape)

        return image[:,:3]

def pytorch3d_rasterize(vertices, faces, image_size, attributes=None, 
                        soft=False, blur_radius=0.0, sigma=1e-8, faces_per_pixel=1, gamma=1e-4, 
                        perspective_correct=False, clip_barycentric_coords=True, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]

        if h is None and w is None:
            image_size = image_size
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h

        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        # import ipdb; ipdb.set_trace()
        # pytorch3d rasterize
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            perspective_correct=perspective_correct,
            clip_barycentric_coords=clip_barycentric_coords,
            # max_faces_per_bin = faces.shape[1],
            bin_size = 0
        )
        # import ipdb; ipdb.set_trace()
        vismask = (pix_to_face > -1).float().squeeze(-1)
        depth = zbuf.squeeze(-1)
        
        if soft:
            from pytorch3d.renderer.blending import _sigmoid_alpha
            colors = torch.ones_like(bary_coords)
            N, H, W, K = pix_to_face.shape
            pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
            pixel_colors[..., :3] = colors[..., 0, :]
            alpha = _sigmoid_alpha(dists, pix_to_face, sigma)
            pixel_colors[..., 3] = alpha
            pixel_colors = pixel_colors.permute(0,3,1,2)
            return pixel_colors

        if attributes is None:
            return depth, vismask
        else:
            vismask = (pix_to_face > -1).float()
            D = attributes.shape[-1]
            attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
            N, H, W, K, _ = bary_coords.shape
            mask = pix_to_face == -1
            pix_to_face = pix_to_face.clone()
            pix_to_face[mask] = 0
            idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
            pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
            pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
            pixel_vals[mask] = 0  # Replace masked values in output.
            pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
            pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
            return pixel_vals