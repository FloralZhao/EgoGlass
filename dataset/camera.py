import os
import numpy as np
from . import config as dconfig

# ------------------- project -------------------
def project_point_distortion(P, R, T, f, c, k, p):  # for a single frame
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion

    Args
      P: Nx3 points in world coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
      f: 2x1 Camera focal length
      c: 2x1 Camera center
      k: 3x1 Camera radial distortion coefficients
      p: 2x1 Camera tangential distortion coefficients
    Returns
      Proj: Nx2 points in pixel space
      D: 1xN depth of each point in camera space
      radial: 1xN radial distortion per point
      tan: 1xN tangential distortion per point
      r2: 1xN squared radius of the projected points before distortion
    """
    assert len(P.shape)==2
    assert P.shape[1]==3

    N = P.shape[0]  # num_joints
    X = R.dot(P.T)+T  # rotate and translate (3, N)
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2  # (N, )

    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2


def project_point_no_distortion(P, R, T, f, c, k, p):  # This is for a single frame
  """
  Project points from 3d to 2d using camera parameters
  no distortion correction

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: 2x1 Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  N = P.shape[0]  # num_joint
  X = R.dot(P.T) + T  # rotate and translate  (3, N)
  XX = X[:2, :] / X[2, :]
  r2 = XX[0, :]**2 + XX[1, :]**2

  radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2**2, r2**3]))
  tan = p[0]*XX[1, :] + p[1]*XX[0, :]

  # XXX = XX * np.tile(radial+tan, (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

  Proj = (f * XX) + c
  Proj = Proj.T

  D = X[2, ]

  return Proj, D, radial, tan, r2



# ------------------- coordinate system -------------------
def world_to_camera_frame(P, R, T):
    """
    Convert points from world to camera coordinates
    3D to 3D

    Args
      P: num_joints *3 3d points in world coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
    Returns
      X_cam: num_joints *3 3d points in camera coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.dot(P.T) + T  # rotate and translate

    return X_cam.T


def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame
  3D to 3D

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.T.dot(P.T - T)  # rotate and translate

  return X_cam.T


# ------------------- Load camera parameters -------------------
def load_camera_intrinsic_params(filename, cam_type='bodycam'):
    """Load camera intrinsic parameters for both body cameras and PointGreys
      Args
          filename: file with cameras data
          cam_type: type of the cameras, {'bodycam', 'pgcam'}

    Returns
        f: 2x1 Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    """
    with open(filename) as file:
        lines = file.readlines()
    f = np.zeros((2, 1))
    f[0] = float(lines[1].split()[0])
    f[1] = float(lines[2].split()[1])

    c = np.zeros((2, 1))
    c[0] = float(lines[1].split()[2])
    c[1] = float(lines[2].split()[2])

    if cam_type == 'bodycam':
        line_id = 4
    else:
        line_id = 10

    k = np.zeros((3, 1))
    k[0] = float(lines[line_id].split()[0])
    k[1] = float(lines[line_id+1].split()[0])
    k[2] = float(lines[line_id+4].split()[0])

    p = np.zeros((2, 1))
    p[0] = float(lines[line_id+2].split()[0])
    p[1] = float(lines[line_id+3].split()[0])

    return f, c, k, p

def load_camera_extrinsic_params(filename):
    ''' Load extrinsics for PG cameras
    :return:
        R: np.array. 3*3 rotation matrix
        T: np.array 3*1 translation vector
    '''

    with open(filename) as file:
        lines = file.readlines()
    line = ''.join(lines[4:7]).split()
    R = np.array([float(x) for x in line]).reshape([3,3])
    T = np.reshape(np.array([float(x) for x in lines[7:10]]), (3, 1))

    return R, T



def load_bcams(calib_dir='/net/vision29/data/ssd/dongxuz1/EgoDataset/BodycamCalib'):
    ''' load instrinsics of body cameras'''
    num_cam = dconfig.BODYCAMS_NUM
    bcams = {}
    for cam in range(num_cam):
        filename = os.path.join(calib_dir, f'BodyCam{cam+1}.ini')
        bcams[cam+1] = load_camera_intrinsic_params(filename)

    return bcams  # f, c, k, p


def load_pcams(calib_dir='/net/vision29/data/ssd/dongxuz1/EgoDataset/PGCalib'):
    ''' Load both intrinsics and extrinsics for PG cameras'''
    num_cam = dconfig.PGCAMS_NUM
    pcams = {}
    for cam in range(num_cam):
        filename = os.path.join(calib_dir, f'calib_{dconfig.PG[cam]}.txt')
        f, c, k, p = load_camera_intrinsic_params(filename, 'pgcam')
        R, T = load_camera_extrinsic_params(filename)
        pcams[cam+1] = f, c, k, p, R, T

    return pcams


# ------------------- Load tracking -------------------
def load_tracking(root_dir='/net/vision29/data/ssd/dongxuz1/EgoDataset', subjects=None, actions=None):
    """For body cameras, their R, T and valid or not

    Args
      root_dir: path of data directory
      subjects: the subjects id
      actions: actions to load
    Returns
      rcams: dictionary of R and T for (subject, action, bcam)
      isValid:  a dictionary of whether each frame is valid or not for (subject, action)
      R: num_valid_frame*3*3
      T: num_valid_frame*3*1
      valid: num_frame
    """

    if subjects is None:
        subjects = [1,2,3,4,5,6,7,8,9,10]
    if actions is None:
        actions = ['sitting', 'standing', 'walking']

    num_cam = dconfig.BODYCAMS_NUM
    rcams = {}
    isValid = {}

    for id in subjects:
        for action in actions:
            file = os.path.join(root_dir, f'S{id}', action, 'bodyCamTrackFile.txt')

            with open(file) as f:
                lines = f.readlines()
            num_frame = int(lines[0].split()[1])
            valid = np.ones(num_frame, dtype=bool)
            R = np.zeros((num_cam, num_frame, 3, 3))
            T = np.zeros((num_cam, num_frame, 3, 1))

            line_id = 2
            idx = 0  # number of invalid frames
            while line_id < len(lines):
                frame_id = int(lines[line_id].split()[1])
                valid[frame_id] = bool(int(lines[line_id + 1].split()[1]))
                line_id += 2

                if not valid[frame_id]:
                    idx += 1  # one more invalid frame
                    R = np.delete(R, -1, axis=1)
                    T = np.delete(T, -1, axis=1)
                    continue
                for cam_id in range(num_cam):
                    line = ''.join(lines[line_id:line_id + 3]).split()
                    cam_mat = np.array([float(x) for x in line]).reshape(3, 4)
                    R[cam_id, frame_id - idx, :, :] = cam_mat[:, :-1]
                    T[cam_id, frame_id - idx, :, :] = cam_mat[:, -1].reshape(3, 1)
                    line_id += 4

            assert R.shape[1]+idx == num_frame, f"frame number mismatch for {id}/{action}"

            isValid[(id, action)] = valid
            for i in range(num_cam):
                rcams[(id, action, i+1)] = R[i], T[i]

    return rcams, isValid


if __name__ == '__main__':
  bcams = load_bcams()
  pcams = load_pcams()
  rcam, isvalid = load_tracking(subjects=[3], actions=['sitting'])