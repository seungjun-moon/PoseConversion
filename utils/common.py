import os
import pickle
from utils.rotation_converter import *
from pytorch3d.transforms import matrix_to_euler_angles, matrix_to_axis_angle

def pickle_dump(loadout, file):
    '''
    Dump a pickle file. Create the directory if it does not exist.
    '''
    os.makedirs(os.path.dirname(str(file)), exist_ok=True)

    with open(file, 'wb') as f:
        pickle.dump(loadout, f)

def batch_euler2matrix(pose):
    '''
    Input : Pose Tensor [N * J * 3]

    Output : Pose Tensor [N * J * 3 * 3]
    '''

    out = torch.empty((pose.shape[0], pose.shape[1], pose.shape[2], pose.shape[2]))

    for i in range(len(pose)): #N
        out[i] = batch_rodrigues(batch_euler2axis(pose[i]))

    return out

def batch_matrix2euler(pose):
    '''
    Input : Pose Tensor [N * J * 3 * 3]

    Output : Pose Tensor [N * J * 3]
    '''
    return matrix_to_euler_angles(pose, convention="XYZ")

def batch_matrix2axis(pose):
    '''
    Input : Pose Tensor [N * J * 3 * 3]

    Output : Pose Tensor [N * J * 3]
    '''
    return matrix_to_axis_angle(pose)
