import torch
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(sample['image'].shape)<3:
            sample['image'] = sample['image'][..., None]
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        if len(sample['label'].shape)<3:
            sample['label'] = sample['label'][..., None]
        mask = np.array(sample['label']).astype(np.uint8).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        sample['image'] = img
        sample['label'] = torch.cat((1-mask, mask), dim=0)
        return sample

class ToTensor_3d(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: D0 x H x W
        img = np.array(sample['image']).astype(np.float32)  # D0 x H x W
        mask = np.array(sample['label']).astype(np.uint8)   # D0 x H x W

        # normalization
        max_value = np.max(img)
        min_value = np.min(img)
        img = (img - min_value) / (max_value - min_value)

        mask = mask > 0

        # remove silces without labels, according to the experiment settings in SAML (https://arxiv.org/abs/2007.02035)
        img_first, img_last = img[0,...][None], img[-1,...][None]
        img = np.concatenate((img_first, img, img_last), axis=0)

        mask_first, mask_last = np.zeros([1,mask.shape[1],mask.shape[2]]), np.zeros([1,mask.shape[1],mask.shape[2]])
        mask = np.concatenate((mask_first, mask, mask_last), axis=0)

        non_zero_idx = np.nonzero(np.sum(mask, axis=(1,2)))[0]
        img_crop = img[(non_zero_idx[0]-1):(non_zero_idx[-1]+2), ...]

        img_crop = torch.from_numpy(img_crop).float().unsqueeze(0)  # 1 x (D1+2) x H x W         D1=non_zero_idx[1]-non_zero_idx[0] +1
        mask = torch.from_numpy(mask).float().unsqueeze(0)          # 1 x (D0+2) x H x W
        sample['image'] = img_crop
        sample['label'] = torch.cat([1-mask, mask], dim=0)          # 2 x (D0+2) x H x W
        sample['non_zero_idx'] = [non_zero_idx[0], non_zero_idx[-1]+1]
        return sample
