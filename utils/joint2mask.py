import numpy as np
import cv2
from skimage.transform import resize
import pdb


I = np.array([7, 8, 10, 11, 1, 2, 4, 5])  # start points
J = np.array([8, 9, 11, 12, 2, 3, 5, 6])  # end points

# colors = [[0, 0, 255], [0, 0, 255],  # R leg
#           [0, 255, 0],[0, 255, 0],  # L leg
#           [0, 255, 255], [0, 255, 255],  # R arm
#           [255, 0, 0], [255, 0, 0]]  # L arm

colors = [[0.733, 0, 0], [0.733, 0, 0], # L arm
          [0, 0.733, 0], [0, 0.733, 0], # R arm
          [0.733, 0.733, 0], [0.733, 0.733, 0], # L leg
          [0, 0, 0.733], [0, 0, 0.733]]  # R leg

color2index = {
        (0, 0, 0) : 0,
        (0, 0, 255) : 1,
        (0, 255, 255) : 2,
        (0, 255, 0) : 3,
        (255, 0, 0) : 4,
    }



def im2index(im):
    '''
    turn a 3 channel RGB image to 1 channel index image
    '''
    assert len(im.shape)==3
    height, width, ch = im.shape
    assert ch==3
    m_label = np.zeros((height, width, 1), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            b, g, r = im[h, w, :]
            m_label[h, w, :] = color2index[(b, g, r)]
    return m_label



def joint2mask(mask_size, img_size, pose_2d, mask_thickness, img_name=None):
    '''
    Using OpenCV
    '''

    # ==================== this function do not change the value of pose_2d ===================

    pose_2d = np.reshape(pose_2d.copy(), (-1, 2)).astype(np.int)
    feat_stride = np.array(img_size)/np.array(mask_size)

    valid_joints = []
    for p2d in pose_2d:
        p2d[0] = int(p2d[0]/feat_stride[1]+0.5)
        p2d[1] = int(p2d[1]/feat_stride[0]+0.5)
        if p2d[0] >= 0 and p2d[0] < mask_size[1] and p2d[1] >= 0 and p2d[1] < mask_size[0]:
            valid_joints.append(True)
        else:
            valid_joints.append(False)
    mask = np.zeros(mask_size+[3,])
    for bone in range(len(I)):
        if valid_joints[I[bone]] or valid_joints[J[bone]]:
            mask = cv2.line(mask, tuple(pose_2d[I[bone]]),
                            tuple(pose_2d[J[bone]]),
                            colors[bone], thickness=int(mask_thickness))
    if img_name is not None:
        cv2.imwrite(img_name, mask)
    # mask = im2index(mask)
    return mask