import os
import cv2
import numpy as np
import torchvision
import torch.nn.functional as F

def visualize_grid(visdict, savepath=None, size=512, dim=2, return_grid=True, print_key=True, vis_keys=None):
    '''
    image range should be [0,1]
    dim: 2 for horizontal. 1 for vertical
    '''
    assert dim == 1 or dim==2
    grids = {}
    if dim==2:
        n_row = 1
    else:
        n_row = 8
    if vis_keys is None:
        vis_keys = visdict.keys()
    for key in vis_keys:
        _,_,h,w = visdict[key].shape
        if dim == 2:
            new_h = size; new_w = int(w*size/h)
        elif dim == 1:
            new_h = int(h*size/w); new_w = size
        grid_image = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu(), nrow=n_row)
        grid_image = (grid_image.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        grid_image = np.ascontiguousarray(grid_image)
        # cv2.wri
        if print_key:
            grid_image = cv2.putText(grid_image, key, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.8, [255,0,0], 2, cv2.LINE_AA)
        grids[key] = grid_image
    grid_image = np.concatenate(list(grids.values()), axis=dim-1)
    if savepath:
        cv2.imwrite(savepath, grid_image)
    if return_grid:
        return grid_image

def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    # assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    channels = vertices.shape[-1]

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]

    if vertices.shape[0] == 1:
        fv =  vertices[0][faces[0],:].unsqueeze(0)
        return fv

    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, channels))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]