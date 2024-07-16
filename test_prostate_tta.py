#!/usr/bin/env python

import argparse
import os
import os.path as osp
import tqdm
from dataloaders import dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from dataloaders import utils
from utils.Utils import _connectivity_region_analysis, save_per_img_prostate
from utils.metrics import *
from datetime import datetime
import pytz
from networks.deeplabv3 import *
import cv2
import numpy as np
from medpy.metric import binary
import SimpleITK as sitk

def construct_color_img(prob_per_slice):
    shape = prob_per_slice.shape
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = prob_per_slice * 255
    img[:, :, 1] = prob_per_slice * 255
    img[:, :, 2] = prob_per_slice * 255

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return im_color


def normalize_ent(ent):
    '''
    Normalizate ent to 0 - 1
    :param ent:
    :return:
    '''
    max = np.amax(ent)
    # print(max)

    min = np.amin(ent)
    # print(min)
    return (ent - min) / 0.4


def draw_ent(prediction, save_root, name):
    '''
    Draw the entropy information for each img and save them to the save path
    :param prediction: [2, h, w] numpy
    :param save_path: string including img name
    :return: None
    '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    # save_path = os.path.join(save_root, img_name[0])
    smooth = 1e-8
    cup = prediction[0]
    disc = prediction[1]
    cup_ent = - cup * np.log(cup + smooth)
    disc_ent = - disc * np.log(disc + smooth)
    cup_ent = normalize_ent(cup_ent)
    disc_ent = normalize_ent(disc_ent)
    disc = construct_color_img(disc_ent)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup_ent)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)


def draw_mask(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    cup = prediction[0]
    disc = prediction[1]

    disc = construct_color_img(disc)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)



def draw_boundary(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'boundary')):
        os.makedirs(os.path.join(save_root, 'boundary'))
    boundary = prediction[0]

    boundary = construct_color_img(boundary)
    cv2.imwrite(os.path.join(save_root, 'boundary', name.split('.')[0]) + '.png', boundary)


def adjust_conv(output, uncertainty, feat, num_class):
    uncertainty = (uncertainty<0.2) * uncertainty
    temp_pred = torch.argmax(output, dim=1, keepdim=True)  # D1x1xHxW
    weight_volume = []
    for c in range(num_class):
        uncertainty_temp = F.interpolate((temp_pred == c) * (1 - uncertainty), size=feat.size()[2:], mode="bilinear", align_corners=True)  # D1x1x(H/16)x(W/16)
        weight_volume_temp = F.normalize(torch.mean(uncertainty_temp * feat, dim=(0, 2, 3), keepdim=True), dim=1)  # 1x256x1x1
        weight_volume.append(weight_volume_temp)
    weight_volume = torch.cat(weight_volume, dim=0)
    return weight_volume

def main_test(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    output_path = args.save_path

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.ToTensor_3d()
    ])
    db_test = DL.ProstateSegmentation_val(args, base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                    transform=composed_transforms_test)
    batch_size = 1
    test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=1)

    # 2. model
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))

    checkpoint = torch.load(model_file)
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if args.movingbn:
        model.train()
    else:
        model.eval()

    val_dice = []
    val_asd = []
    val_hd95 = []
    timestamp_start = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    weight = F.normalize(model.state_dict()["decoder.last_conv.weight"], dim=1)

    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),ncols=80, leave=False):
        data = sample['image']                            # 1x1x(D1+2)xHxW
        target = sample['label']                          # 1x2x(D0+2)xHxW
        img_name = sample['img_name']
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        pred_3d = torch.zeros_like(target).cuda()  # 1x2x(D0+2)xHxW
        pred_3d[:, 0, ...] = 1       # background

        data_slice = []
        for s in range(0, data.shape[2]-2):
            slic = data[:, 0, s:s+3, ...]            # 1x3xHxW
            data_slice.append(slic)
        data_slice = torch.cat(data_slice, dim=0)       # Dx3xHxW

        with torch.no_grad():
            output, feat = model(data_slice, None)           # Dx2xHxW

        output = torch.tanh(output)  # D1x2xHxW
        evidence = torch.exp(output / 0.25)  # D1x2xHxW
        alpha = evidence + 1  # D1x2xHxW
        S = torch.sum(alpha, dim=1, keepdim=True)  # D1x1xHxW
        uncertainty = 2 / S  # D1x1xHxW
        output1 = alpha / S  # D1x2xHxW

        # =========================================================================
        # Test Time Adaptation by uncertainty map
        alpha = 0.5
        beta =0.5
        feat = F.normalize(feat, dim=1)  # D1x256x(H/16)x(W/16)
        weight_volume = adjust_conv(output1, uncertainty, feat, 2)

        weight = beta * weight + (1 - beta) * weight_volume if batch_idx > 0 else weight_volume
        output2 = F.conv2d(feat, weight, stride=1, padding=0)  # D1x2x(H/16)x(W/16)
        output2 = F.interpolate(output2, size=output1.size()[2:], mode="bilinear", align_corners=True)  # D1x2xHxW

        output1 = torch.sigmoid(output1)
        output2 = torch.sigmoid(output2)
        output = alpha * output1 + (1 - alpha) * output2
        # =========================================================================

        pred_3d[:, :, sample['non_zero_idx'][0]:sample['non_zero_idx'][1], ...] = output.permute(1, 0, 2, 3).unsqueeze(0)    # 1x2x(D0+2)xHxW

        pred_3d = pred_3d[:, :, 1:-1, ...]                # 1x2xD0xHxW
        target = target[:, :, 1:-1, ...]          # 1x2xD0xHxW

        target_numpy = target[0][1].data.cpu().numpy()                      # D0xHxW
        prediction = torch.argmax(pred_3d, dim=1)[0].data.cpu().numpy()     # D0xHxW
        prediction = _connectivity_region_analysis(prediction)

        dice = binary.dc(prediction, target_numpy)
        hd95 = binary.hd95(prediction, target_numpy) if np.sum(prediction) > 1e-4 else 100
        asd = binary.asd(prediction, target_numpy) if np.sum(prediction) > 1e-4 else 100

        val_dice.append(dice*100)
        val_hd95.append(hd95)
        val_asd.append(asd)

        # save volume
        volume_path = osp.join(output_path, "3d")
        if not osp.exists(volume_path):
            os.makedirs(volume_path)
        out = sitk.GetImageFromArray(prediction)
        sitk.WriteImage(out, osp.join(output_path, "3d", img_name[0].split("\\")[-1]))

        for s in range(0, data.shape[0]-2):
            img = data[s+1, ...][None]
            img = torch.cat([img, img, img], dim=0)
            lt = target_numpy[s + sample['non_zero_idx'][0]-1, ...]
            lp = prediction[s + sample['non_zero_idx'][0]-1, ...]
            img, lt = utils.untransform(img, lt)
            save_img_name = img_name[0].split('.')[0] + "_"+str(s)
            save_per_img_prostate(img.numpy().transpose(1, 2, 0),
                                 output_path,
                                 save_img_name,
                                 lp, lt, mask_path=None, ext="bmp")

    print('\n==>val_avg_dice : {:.4f} + {:.4f}'.format(np.mean(val_dice), np.std(val_dice)))
    print('==>val_avg_hd95   : {:.4f} + {:.4f}'.format(np.mean(val_hd95), np.std(val_hd95)))
    print('==>val_avg_asd    : {:.4f} + {:.4f}'.format(np.mean(val_asd), np.std(val_asd)))
    with open(osp.join(output_path, 'log.csv'), 'a') as f:
        elapsed_time = (
                datetime.now(pytz.timezone('Asia/Hong_Kong')) -
                timestamp_start).total_seconds()
        log = ['batch-size: '] + [batch_size] + [args.model_file] +  \
               ['val_dice: '] + val_dice + \
               ['val_hd95: '] + val_hd95 + \
               ['val_asd: '] + val_asd + [elapsed_time]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='xxxx', help='Model path')
    parser.add_argument('--datasetTest', type=list, default=[1], help='test folder id contain images ROIs to test')
    parser.add_argument('--dataset', type=str, default='test', help='test folder id contain images ROIs to test')
    parser.add_argument('-g', '--gpu', type=int, default=0)

    parser.add_argument('--data-dir', default='../dataset/prostate/', help='data root path')
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+', )
    parser.add_argument('--sync-bn', type=bool, default=False, help='sync-bn in deeplabv3+')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='freeze batch normalization of deeplabv3+')
    parser.add_argument('--movingbn', type=bool, default=False,
                        help='moving batch normalization of deeplabv3+ in the test phase', )
    parser.add_argument('--save_path', type=str, default='../dofe_new2_logs/',
                        help='Path root for test image and mask')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    args = parser.parse_args()

    fold = args.model_file.split("/")
    args.save_path = osp.join(args.save_path, fold[2], fold[3], fold[4], fold[5])   

    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
    main_test(args)
