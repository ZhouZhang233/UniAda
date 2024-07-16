from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import SimpleITK as sitk
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils.data import DataLoader

def get_dataloader(args):

    composed_transforms_tr = transforms.Compose([
        tr.ToTensor()
    ])
    composed_transforms_ts = transforms.Compose([
        tr.ToTensor_3d()
    ])
    domain = ProstateSegmentation(args, base_dir=args.data_dir, phase='train', splitid=args.datasetTrain,
                                     transform=composed_transforms_tr)
    train_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader_all = []
    for test_list in args.datasetTest:
        domain_val = ProstateSegmentation_val(args, base_dir=args.data_dir, phase='test', splitid=[test_list],
                                             transform=composed_transforms_ts)
        val_loader = DataLoader(domain_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        val_loader_all.append(val_loader)

    return train_loader, val_loader_all


class ProstateSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self, args,
                 base_dir='../data/prostate',
                 phase='train',
                 splitid=[2, 3, 4, 5, 6],
                 transform=None,
                 state='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """

        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase

        self.image_pool = {'Domain1': [], 'Domain2': [], 'Domain3': [], 'Domain4': [], 'Domain5': [], 'Domain6': []}
        self.label_pool = {'Domain1': [], 'Domain2': [], 'Domain3': [], 'Domain4': [], 'Domain5': [], 'Domain6': []}
        self.img_name_pool = {'Domain1': [], 'Domain2': [], 'Domain3': [], 'Domain4': [], 'Domain5': [], 'Domain6': []}

        self.splitid = splitid
        # SEED = 1212
        random.seed(args.seed)
        np.random.seed(args.seed)
        for id in splitid:
            self._image_dir = os.path.join(self._base_dir, 'Domain' + str(id), 'image_npy/')

            imagelist = sorted(glob(self._image_dir + '*.npy'))

            print('==> Loading {} data from: {}'.format(phase, self._image_dir))
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path})
                self.image_pool['Domain'+str(id)].append(np.load(image_path))
                self.label_pool['Domain'+str(id)].append(np.load(gt_path))
                self.img_name_pool['Domain'+str(id)].append(image_path.split('/')[-1])

        self.transform = transform

        for k in list(self.image_pool.keys()):
            if not self.image_pool[k]:
                del self.image_pool[k]
                del self.label_pool[k]
                del self.img_name_pool[k]

        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img,
                               'label': np.expand_dims(_target, axis=2),
                               'img_name': _img_name,
                               'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img,
                               'label': np.expand_dims(_target, axis=2),
                               'img_name': _img_name,
                               'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample = anco_sample
        return sample


class ProstateSegmentation_val(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self, args,
                 base_dir='../data/prostate',
                 phase='train',
                 splitid=[2, 3, 4, 5, 6],
                 transform=None,
                 state='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase

        self.image_pool = {'Domain1': [], 'Domain2': [], 'Domain3': [], 'Domain4': [], 'Domain5': [], 'Domain6': []}
        self.label_pool = {'Domain1': [], 'Domain2': [], 'Domain3': [], 'Domain4': [], 'Domain5': [], 'Domain6': []}
        self.img_name_pool = {'Domain1': [], 'Domain2': [], 'Domain3': [], 'Domain4': [], 'Domain5': [], 'Domain6': []}

        self.splitid = splitid
        # SEED = 1212
        random.seed(args.seed)
        np.random.seed(args.seed)
        for id in splitid:
            self._image_dir = os.path.join(self._base_dir, 'Domain' + str(id), 'image_volume/')
            self._gt_dir = os.path.join(self._base_dir, 'Domain' + str(id), 'mask_volume/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = sorted(glob(self._image_dir + '*.nii.gz'))
            gtlist = sorted(glob(self._gt_dir + '*_segmentation.nii.gz'))
            for i, image_path in enumerate(imagelist):
                gt_path = gtlist[i]
                self.image_list.append({'image': image_path, 'label': gt_path})
                itk_image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(itk_image)
                itk_gt = sitk.ReadImage(gt_path)
                gt = sitk.GetArrayFromImage(itk_gt)
                self.image_pool['Domain'+str(id)].append(image)
                self.label_pool['Domain'+str(id)].append(gt)
                self.img_name_pool['Domain'+str(id)].append(image_path.split('/')[-1])

        self.transform = transform

        for k in list(self.image_pool.keys()):
            if not self.image_pool[k]:
                del self.image_pool[k]
                del self.label_pool[k]
                del self.img_name_pool[k]

        print('-----Total number of volumes in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img,
                               'label': np.expand_dims(_target, axis=2),
                               'img_name': _img_name,
                               'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img,
                               'label': _target,
                               'img_name': _img_name,
                               'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample = anco_sample
        return sample

