import torch
from torchvision import transforms
import numpy as np

class Rescale:
    """ Use before ToTensor()
    Only rescale bimages"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        bimages = sample['bimages']   # list of 6 PIL images
        # Expects PIL Image
        t = transforms.Resize(self.output_size)
        if isinstance(bimages, list):
            sample['bimages'] = [t(bimage) for bimage in bimages]
        else:
            sample['bimages'] = t(bimages)
        return sample

class MaskTrsf:
    def __call__(self, sample):
        if 'mask' not in list(sample.keys()):
            return sample
        mask = sample['mask']
        if isinstance(mask, list):
            mask = [np.transpose(m, [2,0,1]) for m in mask]
        else:
            mask = np.transpose(mask, [2,0,1])
        sample['mask'] = mask
        return sample


class ToTensor:
    """ Convert ndarrays in sample to Tensors"""
    def __call__(self, sample):
        bimages = sample['bimages']
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        if isinstance(bimages, list):
            sample['bimages'] = [transforms.ToTensor()(bimage) for bimage in bimages]
        else:
            sample['bimages'] = transforms.ToTensor()(bimages)

        keys = list(sample.keys())
        for k in keys:
            if k == 'bimages':
                continue
            elif isinstance(sample[k], list):
                k_tensor = []
                for element in sample[k]:
                    k_tensor.append(torch.from_numpy(element).float())
                sample.update({k: k_tensor})
            else:
                pytorch_data = torch.from_numpy(sample[k]).float()
                sample.update({k: pytorch_data})

        # sample['pose_3d'] = torch.from_numpy(sample['pose_3d'])
        # sample['pose_2d'] = torch.from_numpy(sample['pose_2d'])
        # sample['heatmap'] = torch.from_numpy(sample['heatmap'])

        return sample

class Normalize:
    '''
    Normalize images
    Expects tensors of C*H*W
    '''

    def __init__(self, mean, std, view_id=None, inplace=False):
        '''
        :param mean: a dict of means for both bimages and pimages
        :param std: a dict of std for both bimages and pimages
        '''
        self.mean = mean
        self.std = std
        if view_id is not None:
            self.view_id = view_id
        self.inplace = inplace

    def __call__(self, sample):
        bimages_mean = self.mean['BIMAGES_MEAN']
        bimages_std = self.std['BIMAGES_STD']

        bimages = sample['bimages']

        if isinstance(bimages, list):
            bcams_num = len(bimages)
            for i in range(bcams_num):
                bnorm = transforms.Normalize(mean=bimages_mean[i], std=bimages_std[i], inplace=self.inplace)
                bimages[i] = bnorm(bimages[i])
        else:
            bimages = transforms.Normalize(mean=bimages_mean[self.view_id], std=bimages_std[self.view_id], inplace=self.inplace)(bimages)
        sample['bimages'] = bimages
        return sample


# ------------------- Fuse 6 images -------------------
class ConcatBimages:
    ''' Concat 6 images along a dim'''
    def __init__(self, dim=0):
        self.dim = dim
    def __call__(self, sample):
        bimages = sample['bimages']  # a list of 6 elements. Each element is a tensor of shape [3, height, width]
        cbimages = torch.cat(bimages, dim=self.dim)
        sample['bimages'] = cbimages

        return sample

class StackBimages:
    '''Concatenates 6 images along a new dimension.'''
    def __init__(self, dim=0):
        self.dim = dim
    def __call__(self, sample):
        bimages = sample['bimages']
        sbimages = torch.stack(bimages, dim=self.dim)  # (cam_num, 3, height, width). After mini-batch, (batch_size, cam_num, 3, height, width)
        sample['bimages'] = sbimages

        return sample
