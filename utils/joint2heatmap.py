import numpy as np
import cv2
import skimage.io as sio
from skimage.transform import resize as resize



def joint2heatmap(p2d, cfg, heatmap_type='gaussian'):
    '''
    Args:
        joints: [num_joints, 3]
        heatmap_type

    Returns:
        visible(1: visible, 0: not visible)


    *********** NOTE: this function will change the value of p2d, make copy before using it **********

    '''
    num_joints = p2d.shape[0]
    sigma = cfg.DATA.SIGMA
    heatmap_size = (int(cfg.MODEL.HEATMAP_SIZE[0]), int(cfg.MODEL.HEATMAP_SIZE[1]))
    image_size = cfg.MODEL.IMAGE_SIZE
    ori_body_img_size = cfg.DATA.ORI_BODY_IMG_SIZE

    visible = np.ones((num_joints), dtype=np.float32)

    assert heatmap_type == 'gaussian', 'Only support gaussian map now!'


    # don't change the value of p2d here
    # p2d[:, 0] /= (int(ori_body_img_size[1]) / heatmap_size[1])
    # p2d[:, 1] /= (int(ori_body_img_size[0]) / heatmap_size[0])
    # p2d = p2d.astype(np.int)



    if heatmap_type == 'gaussian':
        heatmaps = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
        tmp_size = sigma*3


        for joint in range(num_joints):
            feat_stride = np.array(ori_body_img_size) / np.array(heatmap_size)
            mu_x = int(p2d[joint][0] / feat_stride[1] + 0.5)
            mu_y = int(p2d[joint][1] / feat_stride[0] + 0.5)


            # if mu_x < 0 or mu_x >= image_size[1] or mu_y < 0 or mu_y>=image_size[0]:
            #     visible[joint] = 0
            #     continue

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= heatmap_size[1] or ul[1] >= heatmap_size[0] or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                visible[joint] = 0
                continue

            # generate Gaussian
            sz = 2 * tmp_size + 1
            x = np.arange(0, sz, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = sz // 2
            # The gaussian is not normalized, we want the cneter value to equal 1
            g = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[0])

            heatmaps[joint][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return heatmaps, visible



def generate_visible(p2d, cfg):
    '''
    Args:
        joints: [num_joints, 3]

    Returns:
        visible(1: visible, 0: not visible)


    *********** NOTE: this function will change the value of p2d **********

    '''
    num_joints = cfg.DATA.NUM_JOINTS
    visible = np.ones((num_joints), dtype=np.float32)
    heatmap_size = (int(cfg.MODEL.HEATMAP_SIZE[0]), int(cfg.MODEL.HEATMAP_SIZE[1]))
    image_size = cfg.MODEL.IMAGE_SIZE
    ori_body_img_size = cfg.DATASET.PARAMS.ORI_BODY_IMG_SIZE

    tmp_size = cfg.MODEL.PARAMS.SIGMA*3


    for joint in range(num_joints):
        feat_stride = np.array(ori_body_img_size[0:-1]) / np.array(heatmap_size)
        mu_x = int(p2d[joint][0] / feat_stride[1] + 0.5)
        mu_y = int(p2d[joint][1] / feat_stride[0] + 0.5)


        if mu_x < 0 or mu_x >= image_size[1] or mu_y < 0 or mu_y>=image_size[0]:
            visible[joint] = 0
            continue

        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        if ul[0] >= heatmap_size[1] or ul[1] >= heatmap_size[0] or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            visible[joint] = 0



    return visible
