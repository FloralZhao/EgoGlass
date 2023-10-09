'''
w/ heatmap & visible
heatmat & visible are list of num_cam
'''
import os
import numpy as np
import itertools
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.io as sio
from PIL import Image
import random

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.joint2heatmap import joint2heatmap
from utils.joint2mask import joint2mask
from . import camera
from . import data_utils
from . import config as dconfig
from . import transform_utils
from . import transform2D
from preprocess import process_all

# import sys
# sys.path.append("../")
# from utils.joint2heatmap import joint2heatmap
# import camera
# import data_utils
# from config import config as dconfig
# import transform_utils

import pdb

class EgoDataset(Dataset):
    def __init__(self, cfg, isTrain, transform=None):
        '''
        :param cfg:
        :param isTrain: bool, TrainSet or TestSet
        :param transform: default is None
        '''

        # ------------------- Basics -------------------
        self.root_dir = dconfig.ROOT_DIR
        self.cfg = cfg
        self.augment = cfg.DATA.AUGMENT
        self.cm_to_m = cfg.DATA.CM_TO_M
        if isTrain:
            self.subjects = sorted(cfg.DATA.TRAIN_SUBJECTS)  # in ascending order
            self.actions = sorted(cfg.DATA.TRAIN_ACTIONS)  # in the order of ['sitting', 'standing', 'walking']
        else:
            self.subjects = sorted(cfg.DATA.TEST_SUBJECTS)
            self.actions = sorted(cfg.DATA.TEST_ACTIONS)

        self.flip_pairs = dconfig.FLIP_PAIRS

        if transform is None:
            self.transform = transforms.Compose([transform_utils.Rescale(cfg.MODEL.IMAGE_SIZE),
                                                 transform_utils.MaskTrsf(),
                                                 transform_utils.ToTensor(),
                                                 transform_utils.Normalize(mean=dconfig.MEAN, std=dconfig.STD),
                                                 ])
        else:
            self.transform = transform

        self._subjects_str = list(map(str, self.subjects))
        self._subjects_str = list(map((lambda x: 'S' + x), self._subjects_str))
        self.name = '_'.join(self._subjects_str) + \
                    ''.join(list(map((lambda x: '_' + x), self.actions)))
        self.cam_num = cfg.DATA.BODYCAMS_NUM

        self.bcams = camera.load_bcams(os.path.join(self.root_dir, 'BodycamCalib'))
        # self.pcams = camera.load_pcams(os.path.join(self.root_dir, 'PGCalib'))

        # return a dictionary of frame numbers
        # self._frame_nums = self._get_frame_nums()

        self.img_list = self._get_img_list()

        self.rcams_set, valid_set = camera.load_tracking(self.root_dir, subjects = self.subjects, actions = self.actions)


        # load pose data
        self.pose_2d_dict, self.pose_3d_dict = self._load_pose(valid_set)

        # make array from dict
        self.pose_2d_mat, self.pose_3d_mat, self.rcams_mat_R, self.rcams_mat_T, self.root_positions = self._dict_to_mat()

        if self.cfg.DATA.FILTER_POSE:
            self._filterPose()

        # make ground truth for augmented data
        if self.augment:
            self._augment()


    def __len__(self):
        return self.pose_3d_mat.shape[0]

    def __getitem__(self, item):
        '''
        :param item:
        :return:
        '''
        if torch.is_tensor(item):
            item = item.tolist()

        p3d = self.pose_3d_mat[item].copy()
        p2d = self.pose_2d_mat[item].copy()
        img_path = self.img_list[item]
        action = img_path.split('/')[1]
        root_position = self.root_positions[item]
        bcam_R = self.rcams_mat_R[item]
        bcam_T = self.rcams_mat_T[item]


        bimages = []
        for bcam in self.cfg.DATA.BODYCAMS:
            im = Image.open(os.path.join(self.root_dir, img_path.replace('BodyCam1', bcam)))
            bimages.append(im)

        p2d = p2d.reshape((self.cam_num, -1, 2))


        # ------------------ transformation augmentation ------------------
        # if self.cfg.DATA.AUGMENT:
        #     flipping
            # if random.random()>0.5:
            #     for i in range(self.cam_num):
            #         bimages[i] = bimages[i][:,::-1,:]
            #         p2d[i] = transform2D.flip_pairs(p2d[i], bimages[i].shape[1], self.flip_pairs)

            # cropping and rotating
            # sf = self.cfg.SCALE_FACTOR
            # rf = self.cfg.ROC_FACTOR
            # s = np.clip(np.random.randn()*sf+1, 1-sf, 1+sf)
            # r = np.clip(np.random.randn()*rf, -rf*2, rf*2) if random.random() <= 0.6 else 0
            #
            # c = [dconfig.BODY_IMG_SIZE[1]/2, dconfig.BODY_IMG_SIZE[0]/2]
            # trans =


        heatmaps = []
        visibles = []
        masks = []
        for i in range(self.cam_num):
            heatmap, visible = joint2heatmap(p2d[i][1:], self.cfg)
            heatmaps.append(heatmap)
            visibles.append(visible)
            mask = joint2mask(self.cfg.MODEL.MASK_SIZE, (480, 640), p2d[i], self.cfg.MODEL.MASK_THICKNESS)
            masks.append(mask)
        # heatmaps = np.stack(heatmaps, axis=0)  # (cam_num, num_joints, heatmap_size[0], heatmap_size[1])
        # visibles = np.stack(visibles, axis=0)  # (cam_num, num_joints)

        sample = {'bimages': bimages,
                  'pose_3d': p3d,
                  'pose_2d': p2d,
                  'heatmap': heatmaps,
                  'visible': visibles,
                  'mask': masks
                  }

        if self.transform is not None:
            sample = self.transform(sample)

        bimages = sample['bimages']
        # https://discuss.pytorch.org/t/torch-dataloader-gives-torch-tensor-as-ouput/31084
        p2d = sample['pose_2d']
        p3d = sample['pose_3d'].reshape((-1, 3))
        if self.cm_to_m:
            p3d /= 100

        heatmaps = sample['heatmap']
        visible = sample['visible']
        mask = sample['mask']

        return bimages, p2d, p3d, heatmaps, visible, img_path, action, root_position, bcam_R, bcam_T, mask


    def read_frame_nums(self):
        ''' return a dictionary of valid frame numbers of each (subject, action) pair from a pre-defined file '''
        filename = os.path.join(self.root_dir, 'valid_frame_nums.txt')
        frame_nums = {}

        with open(filename) as f:
            lines = f.readlines()

        line_id = 0
        while line_id < len(lines):
            sbj = int(lines[line_id][1:])
            frame_nums[(sbj, lines[line_id + 1].split()[0])] = int(lines[line_id + 1].split()[1])
            frame_nums[(sbj, lines[line_id + 2].split()[0])] = int(lines[line_id + 2].split()[1])
            frame_nums[(sbj, lines[line_id + 3].split()[0])] = int(lines[line_id + 3].split()[1])
            line_id = line_id + 4

        return frame_nums


    def _get_frame_nums(self):
        frame_nums_full = self.read_frame_nums()
        frame_nums = {}
        # filter subjects and actions
        for existing_key in itertools.product(self.subjects, self.actions):
            frame_nums[existing_key] = frame_nums_full[existing_key]
        return frame_nums


    def _get_img_list(self):
        img_names_all = []
        for s_id in self.subjects:
            for a in self.actions:
                file = os.path.join(self.root_dir, f"S{s_id}", a, 'img_list.npy')
                img_list = np.load(file)
                for img in img_list:
                    img_names_all.append(os.path.join(f"S{s_id}", a, 'TrackingData', 'BodyCam1', img))
        return img_names_all


    def _get_img_list_aug(self, img_list):
        img_names_all = []
        for img_path in img_list:
            for aug in range(dconfig.AUG_NUM):
                img_names_all.append(os.path.join("/".join(img_path.split("/")[:3]), 'AugmentedData/BodyCam1',
                                                  img_path.split("/")[-1].split(".")[0]+"_%03d.jpg"%aug))

        return img_names_all


    def _load_pose(self, valid_set):
        '''
        Load both 3d pose and 2d pose in dict
        :return:
        pose_3d_dict: dict
        pose_2d_dict: dict
        '''

        # read 3d data
        pose_3d_dict_full = data_utils.load_pose_data(self.root_dir, valid_set, self.subjects, self.actions, dim=3)
        pose_3d_dict_full_cam = data_utils.change_coordinate(pose_3d_dict_full, self.rcams_set)  # in the coordinate system of BodyCam1

        # zero-center 3d data
        # pose_3d_dict_full_cam_center = copy.deepcopy(pose_3d_dict_full_cam)
        pose_3d_dict_full_cam_center, self.root_positions_dict = data_utils.centralize(pose_3d_dict_full_cam)

        # project to get 2d data
        pose_2d_dict_full = data_utils.project_to_cameras(pose_3d_dict_full, self.rcams_set, self.bcams)  # (sbj,act,cam)

        # TODO: normalize (2d and 3d)??

        # concat six sets of 2d pose positions for each frame
        pose_2d_dict_full_concat = {}
        for key3d in sorted(pose_3d_dict_full_cam_center.keys()):
            s, a = key3d
            pose_set2 = []
            for cam_id in [1,6]:
                pose_set2.append(pose_2d_dict_full[s, a, cam_id])

            pose_set2 = np.hstack(pose_set2)
            pose_2d_dict_full_concat[key3d] = pose_set2

        return pose_2d_dict_full_concat, pose_3d_dict_full_cam_center


    def _dict_to_mat(self):
        # Make array from dict
        pose_3d_mat = np.zeros((len(self.img_list), dconfig.JOINTS_NUM * 3))
        pose_2d_mat = np.zeros((len(self.img_list), dconfig.JOINTS_NUM * 2 * self.cam_num))
        rcams_mat_R = np.zeros((self.cam_num, len(self.img_list), 3, 3))
        rcams_mat_T = np.zeros((self.cam_num, len(self.img_list), 3, 1))
        root_positions = np.zeros((len(self.img_list), 3))

        idx = 0
        for key3d in sorted(self.pose_3d_dict.keys()):
            s, a = key3d
            n3d, _ = self.pose_3d_dict[key3d].shape
            pose_3d_mat[idx:idx + n3d, :] = self.pose_3d_dict[key3d]
            pose_2d_mat[idx:idx + n3d, :] = self.pose_2d_dict[key3d]
            root_positions[idx:idx+n3d, :] = self.root_positions_dict[key3d]
            rcams_mat_R[0, idx:idx+n3d,:,:] = self.rcams_set[(s, a, 1)][0]
            rcams_mat_R[1, idx:idx+n3d,:,:] = self.rcams_set[(s, a, 6)][0]
            rcams_mat_T[0, idx:idx+n3d,:,:] = self.rcams_set[(s, a, 1)][1]
            rcams_mat_T[1, idx:idx+n3d,:,:] = self.rcams_set[(s, a, 6)][1]
            idx += n3d


        rcams_mat_R = np.transpose(rcams_mat_R, (1, 0, 2, 3))
        rcams_mat_T = np.transpose(rcams_mat_T, (1, 0, 2, 3))

        return pose_2d_mat, pose_3d_mat, rcams_mat_R, rcams_mat_T, root_positions




    def _filterPose(self):
        # Rather than include every frame from every video, we can instead wait for the pose to change
        # significantly before storing a new example.
        frame_indices = process_all.select_frame_to_include(self.pose_3d_mat, self.cfg.DATA.FILTER_THRESHOLD)
        self.pose_3d_mat = self.pose_3d_mat[frame_indices]
        self.pose_2d_mat = self.pose_2d_mat[frame_indices]
        self.rcams_mat_R = self.rcams_mat_R[frame_indices]
        self.rcams_mat_T = self.rcams_mat_T[frame_indices]
        self.root_positions = self.root_positions[frame_indices]
        self.img_list = list(np.array(self.img_list)[frame_indices])



    def _augment(self):
        self.pose_2d_mat = np.repeat(self.pose_2d_mat, dconfig.AUG_NUM, axis=0)
        self.pose_3d_mat = np.repeat(self.pose_3d_mat, dconfig.AUG_NUM, axis=0)
        self.rcams_mat_R = np.repeat(self.rcams_mat_R, dconfig.AUG_NUM, axis=0)
        self.rcams_mat_T = np.repeat(self.rcams_mat_T, dconfig.AUG_NUM, axis=0)
        self.root_positions = np.repeat(self.root_positions, dconfig.AUG_NUM, axis=0)

        self.img_list = self._get_img_list_aug(self.img_list)


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    with open('../configFiles/img_2d_3d/xregopose.yaml') as f:
        config = edict(yaml.safe_load(f))

    ds = EgoDataset(config, True)
    assert len(ds.img_list) == ds.pose_3d_mat.shape[0], "dim mismatch"
    dataloader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=1)
    dataiter = iter(dataloader)
    data = next(dataiter)
    print(len(data[0]))  # bimage: list # 6
    print(data[0][0].shape)  # (4, 3, 120, 160)
    print(data[1].shape)  # p2d  # [4, 6, 12, 2]
    print(data[2].shape)  # p3d  # [4, 12, 3]
    print(len(data[3]))  # heatmap: list # 6
    print(data[3][0].shape)  # [4, 12, 32, 40]
    print(len(data[4]))  # visible, list # 6
    print(data[4][0].shape)  # (4, 12)
    print(type(data[5]))  # img_path # tuple
    print(len(data[5]))  # 4  ('S1/walking/TrackingData/AugmentedData/BodyCam1/img_00000_000.jpg', 'S1/walking/TrackingData/AugmentedData/BodyCam1/img_00000_001.jpg', 'S1/walking/TrackingData/AugmentedData/BodyCam1/img_00000_002.jpg', 'S1/walking/TrackingData/AugmentedData/BodyCam1/img_00001_000.jpg')
    print(type(data[6]))  # tuple, 4, ('walking', 'walking', 'walking', 'walking')
    pdb.set_trace()