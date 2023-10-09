import numpy as np
import itertools
import copy
import pdb
import os
from . import camera
from . import config as dconfig

# import dataset.camera as camera
# import dataset.config as dconfig

CM_TO_M = 100

def load_pose_data(root_dir, isValid, subjects=None, actions=None, dim=3):
    """
     Read ground true file
     Load 2d/3d ground truth pose from disk, and put it in an easy-to-acess dictionary

     Args
       root_dir: String. Path where to load the data from (the root dir of all subjects)
       isValid: dictionary
       subjects: List of integers.
       actions: List of strings.
       dim: Integer={2,3}. Load 2 or 3-dimensional data
     Returns:
       data: Dictionary with keys k=(subject, action)
     """
    if subjects is None:
        subjects = [1,2,3,4,5,6,7,8,9,10]
    if actions is None:
        actions = ['sitting', 'standing', 'walking']
    if dim not in [2, 3]:
        raise ValueError('dim must be 2 or 3')

    data = {}
    loaded_seq = 0

    for s in subjects:
        for a in actions:
            loaded_seq += 1
            print(f"Loading subject {s}, action {a}")
            dpath = os.path.join(root_dir, f"S{s}", a)
            print(dpath)

            valid = isValid[s,a]
            skeleton_file = os.path.join(dpath, 'skeletonData.txt')
            with open(skeleton_file) as f:
                lines = f.readlines()
            num_frame = int(lines[0].split()[1])
            num_joints = int(lines[1].split()[1])

            poses = np.zeros((num_frame, dconfig.JOINTS_NUM*dim))
            idx = 0  # number of invalid frames

            line_id = 3
            for frame in range(num_frame):
                if not valid[frame]:
                    poses = np.delete(poses, -1, axis=0)
                    idx += 1  # number of invalid frames
                    line_id += num_joints+1
                    continue
                joint_pos = np.zeros((18, dim))
                for i in range(num_joints):
                    joint_pos[i, :] = np.asarray([float(x) for x in lines[line_id + i].split()[3:3 + dim]])

                # filter out joints that are not used
                joint_pos = joint_pos[dconfig.JOINTS_TO_USE]
                joint_pos = np.reshape(joint_pos, [1, dconfig.JOINTS_NUM * dim])
                poses[frame - idx, :] = joint_pos
                line_id += num_joints+1

            data[(s, a)] = poses
    return data




def centralize(poses_set):
    """
    Center 3d points around root
    Note: poses_set will be changed because dict is mutable in Python

    Args
      poses_set: dictionary with 3d  (will be changed in place)
    Returns
      poses_set: dictionary with 3d data centred around root (RShoulder) joint
      root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = {}
    for k in poses_set.keys():
        # Keep track of the global position
        # change root position index # Here is using the whichever first joint as root
        # here using Nose
        root_positions[k] = copy.deepcopy(poses_set[k][:, 0:3])

        poses = poses_set[k]
        poses = poses - np.tile(poses[:, 0:3], [1, dconfig.JOINTS_NUM])  # Construct an array by repeating A the number of times given by reps.
        poses_set[k] = poses

    return poses_set, root_positions


def project_to_cameras(poses_set, rcams, bcams, ncams=dconfig.BODYCAMS_NUM):
    """
    Project 3d poses to 2d poses of BODY CAMERAS
    3D pose dictionary -> 2D pose dictionary
    (subject, action)  -> (subject, action, bcam)


    Args
      poses_set: dictionary with 3d poses
      bcams: BODY camera intrinsics
      rcams: BODY camera extrinsics
      ncams: number of cameras per subject
    Returns
      t2d: dictionary with 2d poses
    """

    t2d = {}
    for t3dk in sorted(poses_set.keys()):
        s, a = t3dk
        t3d = poses_set[t3dk]

        N = t3d.shape[0]  # num_frame
        for cam in range(ncams):
            f, c, k, p = bcams[cam+1]
            R, T = rcams[(s, a, cam+1)]
            pose_2d = np.zeros((N, dconfig.JOINTS_NUM*2))
            for n in range(N):
                pts2d, _, _, _, _ = camera.project_point_no_distortion(np.reshape(t3d[n, :], (-1, 3)), R[n,:,:], T[n,:,:], f,c,k,p)
                pose_2d[n, :] = np.reshape(pts2d, (1, dconfig.JOINTS_NUM*2))

            t2d[(s, a, cam+1)] = pose_2d
    return t2d


def change_coordinate(poses_set, rcams):
    '''
    Project 3d poses to 3d poses in the coordinate system of body camera
    :param poses_set: a dict of 3d poses (subject, action)
    :param rcams: a dict of body camera extrinsics (subject, action, cam)
    :return:
    a dict of 3d poses (subject, action) but in the coordinate system of a body camera
    '''


    poses_set_cam = {}
    for k in sorted(poses_set.keys()):
        s, a = k
        p3d = poses_set[k]

        N = p3d.shape[0]  # num_frame
        R, T = rcams[(s, a, 1)]  # use the coordinate system of BodyCam1
        p3d_cam = np.zeros((N, dconfig.JOINTS_NUM*3))
        for n in range(N):
            X_cam = camera.world_to_camera_frame(np.reshape(p3d[n, :], (-1, 3)), R[n,:,:], T[n,:,:])
            p3d_cam[n,:] = np.reshape(X_cam, (1, -1))

        poses_set_cam[k] = p3d_cam
    return poses_set_cam
