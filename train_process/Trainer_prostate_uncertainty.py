import time
from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.nn.functional as F
import pytz
from tensorboardX import SummaryWriter

import tqdm
import socket
from utils.metrics import *
from utils.Utils import *
from utils.Utils import _connectivity_region_analysis
from medpy.metric import binary
from utils.e_losses import edl_digamma_loss

bceloss = torch.nn.BCELoss()
celoss = torch.nn.CrossEntropyLoss()
mseloss = torch.nn.MSELoss()
softmax = torch.nn.Softmax(-1)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(object):

    def __init__(self, args, cuda, model, lr, val_loader, train_loader, out, max_epoch, optim, stop_epoch=None,
                 lr_decrease_rate=0.1, interval_validate=None, batch_size=8):
        self.cuda = cuda
        self.model = model
        self.optim = optim
        self.lr = lr
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size
        self.image_size = args.image_size
        self.backbone = args.backbone

        self.val_loaders = val_loader
        self.train_loader = train_loader
        self.timestamp_start = datetime.now()
        self.interval_validate = interval_validate

        self.out = out
        os.makedirs(self.out, exist_ok=True)

        self.log_headers = [
            'epoch',
            'valid/dice',
            'valid/hd95',
            'valid/asd',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1

        self.alpha = args.alpha
        self.beta = args.beta

    def adjust_conv(self, output, uncertainty, feat, num_class):
        uncertainty = (uncertainty<0.2) * uncertainty
        temp_pred = torch.argmax(output, dim=1, keepdim=True)  # D1x1xHxW
        weight_volume = []
        for c in range(num_class):
            uncertainty_temp = F.interpolate((temp_pred == c) * (1 - uncertainty), size=feat.size()[2:], mode="bilinear", align_corners=True)  # D1x1x(H/16)x(W/16)
            weight_volume_temp = F.normalize(torch.mean(uncertainty_temp * feat, dim=(0, 2, 3), keepdim=True), dim=1)  # 1x256x1x1
            weight_volume.append(weight_volume_temp)
        weight_volume = torch.cat(weight_volume, dim=0)
        return weight_volume

    def validate_volume(self):
        training = self.model.training
        self.model.eval()
        num_class = 2
        val_loss = 0
        val_dice = 0
        val_asd = 0
        val_hd95 = 0
        metrics = []
        num_sample = 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                image = sample['image']
                label = sample['label']
                # domain_code = sample['dc']
                B, C, D, H, W = image.shape
                upsample = torch.nn.Upsample(size=(D, self.image_size, self.image_size), mode='trilinear', align_corners=True)
                image = upsample(image)

                data = image.cuda()                                   # 1x1x(D1+2)xHxW
                target_map = label.cuda()                             # 1x2x(D0+2)xHxW
                pred_3d = torch.zeros_like(target_map).cuda()         # 1x2x(D0+2)xHxW
                pred_3d[:, 0, ...] = 1       # background

                data_slice = []
                # get 2.5D images batch
                for s in range(0, data.shape[2]-2):
                    slic = data[:, 0, s:s+3, ...]                           # 1x3xHxW
                    data_slice.append(slic)
                data_slice = torch.cat(data_slice, dim=0)                   # D1x3xHxW

                output, feat = self.model(data_slice, None)      # D1x2xHxW

                output = torch.tanh(output)                     # D1x2xHxW
                evidence = torch.exp(output / num_class)
                alpha = evidence + 1                            # D1x2xHxW
                S = torch.sum(alpha, dim=1, keepdim=True)       # D1x1xHxW
                uncertainty = num_class / S                     # D1x1xHxW
                output = alpha / S                              # D1x2xHxW

                #=========================================================================
                # uncertainty map
                feat = F.normalize(feat, dim=1)                          # D1x256x(H/16)x(W/16)
                weight_volume = self.adjust_conv(output, uncertainty, feat, num_class)
                output2 = F.conv2d(feat, weight_volume, stride=1, padding=0, bias=False)                                                           # D1x2x(H/16)x(W/16)
                output2 = F.interpolate(output2, size=output.size()[2:], mode="bilinear", align_corners=True)                    # D1x2xHxW

                output = torch.sigmoid(output)
                output2 = torch.sigmoid(output2)

                output = self.alpha * output + (1-self.alpha) * output2

                # up-sampling
                upsample = torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
                output = upsample(output)
                #=========================================================================

                pred_3d[:, :, sample['non_zero_idx'][0]:sample['non_zero_idx'][1], ...] = output.permute(1, 0, 2, 3).unsqueeze(0)    # 1x2x(D0+2)xHxW

                pred_3d = pred_3d[:, :, 1:-1, ...]               # 1x2xD0xHxW
                target_map = target_map[:, :, 1:-1, ...]         # 1x2xD0xHxW

                bce_loss = bceloss(pred_3d, target_map)

                loss_seg = bce_loss
                loss_data = loss_seg.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                pred_3d = np.asarray(torch.argmax(pred_3d, dim=1)[0].data.cpu())             # D0xHxW
                target_map = np.asarray(target_map[0][1].data.cpu(), dtype=np.uint8)         # D0xHxW

                pred_3d = _connectivity_region_analysis(pred_3d)

                dice = binary.dc(pred_3d, target_map)
                hd95 = binary.hd95(pred_3d, target_map) if np.sum(pred_3d) > 1e-4 else 100
                asd = binary.asd(pred_3d, target_map) if np.sum(pred_3d) > 1e-4 else 100

                val_dice += dice
                val_hd95 += hd95
                val_asd += asd
                # num_sample += 1
            val_loss /= len(self.val_loader)
            val_dice /= len(self.val_loader)
            val_hd95 /= len(self.val_loader)
            val_asd /= len(self.val_loader)
            metrics.append((val_loss, val_hd95, val_dice, val_asd))
            self.writer.add_scalar('val_data/loss', val_loss, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_dice', val_dice, self.epoch)
            self.writer.add_scalar('val_data/val_hd95', val_hd95, self.epoch)
            self.writer.add_scalar('val_data/val_asd', val_asd, self.epoch)

            print("           dice: {:<13}, hd95: {:<13}, asd: {:<13}".format(np.round(val_dice, 4),
                                                                              np.round(val_hd95, 4),
                                                                              np.round(val_asd, 4)))

            if training:
                self.model.train()
        return [val_dice, val_hd95, val_asd]

    def save_epoch(self, val_dice, val_hd95, val_asd):
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            log = [[self.epoch] + [val_dice] + [val_hd95] + [val_asd]]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_dice = val_dice
        is_best = mean_dice > self.best_mean_dice
        if is_best:
            if self.best_mean_dice != 0:
                os.remove(osp.join(self.out, 'checkpoint_%d_dice_%.4f.pth.tar' % (self.best_epoch,
                                                                                  self.best_mean_dice)))
            self.best_epoch = self.epoch
            self.best_mean_dice = mean_dice
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'learning_rate_gen': get_lr(self.optim),
                'best_mean_dice': self.best_mean_dice,
            }, osp.join(self.out, 'checkpoint_%d_dice_%.4f.pth.tar' % (self.best_epoch, self.best_mean_dice)))
        # else:
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'learning_rate_gen': get_lr(self.optim),
            'best_mean_dice': self.best_mean_dice,
        }, osp.join(self.out, 'checkpoint_latest.pth.tar'))

    def train_epoch(self):
        self.model.train()
        self.running_seg_loss = 0.0
        self.running_bce_loss = 0.0
        self.running_dice_loss = 0.0
        self.running_uncert_loss = 0.0

        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        
        num_class = 2

        for batch_idx, sample in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration

            assert self.model.training
            self.optim.zero_grad()

            image = None
            label = None
            # domain_code = None
            for domain in sample:
                if image is None:
                    image = domain['image']
                    label = domain['label']
                else:
                    image = torch.cat([image, domain['image']], 0)
                    label = torch.cat([label, domain['label']], 0)

            image = image.cuda()
            target_map = label.cuda()
            # down-sampling
            B, C, H, W = image.shape
            upsample = torch.nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
            image = upsample(image)

            output, feat = self.model(image, target_map)

            output = torch.tanh(output)
            evidence = torch.exp(output / num_class)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            uncertainty = num_class / S
            output = alpha / S

            feat = F.normalize(feat, dim=1)                          # D1x256x(H/16)x(W/16)
            weight_volume = self.adjust_conv(output, uncertainty, feat, num_class)

            output2 = F.conv2d(feat, weight_volume, stride=1, padding=0, bias=False)           # D1x2x(H/16)x(W/16)
            output2 = F.interpolate(output2, size=output.size()[2:], mode="bilinear", align_corners=True)      # D1x1xHxW

            output = torch.sigmoid(output)
            output2 = torch.sigmoid(output2)

            # up-sampling
            upsample = torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
            output = upsample(output)
            output2 = upsample(output2)
            evidence = upsample(evidence)

            bce_loss = bceloss(output, target_map) + bceloss(output2, target_map)
            evidence_ce = edl_digamma_loss(evidence, target_map, self.epoch, num_class, 50)

            loss_seg = bce_loss + evidence_ce

            self.running_seg_loss += loss_seg.item()
            self.running_bce_loss += bce_loss.item()
            self.running_uncert_loss += evidence_ce.item()
            loss_data = loss_seg.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss = loss_seg
            loss.backward()
            self.optim.step()

            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(image[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/image', grid_image, iteration)
                grid_image = make_grid(target_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target', grid_image, iteration)
                grid_image = make_grid(output[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/prediction', grid_image, iteration)

            # write loss log
            self.writer.add_scalar('train_gen/loss', loss_data, iteration)
            self.writer.add_scalar('train_gen/loss_seg', loss_seg.data.item(), iteration)

        self.running_seg_loss /= len(self.train_loader)
        self.running_bce_loss /= len(self.train_loader)
        self.running_dice_loss /= len(self.train_loader)
        self.running_uncert_loss /= len(self.train_loader)

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, Average bceLoss: %f, Average uncertLoss: %f' %
              (self.epoch+1, get_lr(self.optim), self.running_seg_loss, self.running_bce_loss, self.running_uncert_loss))

    def train(self):
        for epoch in range(self.epoch, self.max_epoch):
            start_time = timeit.default_timer()
            torch.cuda.empty_cache()
            self.epoch = epoch
            # training
            self.train_epoch()

            if (epoch + 1) % (self.max_epoch // 2) == 0:
                self.lr = self.lr * self.lr_decrease_rate
                for param_group in self.optim.param_groups:
                    param_group['lr'] = self.lr

            self.writer.add_scalar('lr', get_lr(self.optim), self.epoch * (len(self.train_loader)))

            # validation
            if (self.epoch + 1) % self.interval_validate == 0 or self.epoch == 0:
                dice_avg, hd95_avg, asd_avg = 0,0,0
                for val in self.val_loaders:
                    self.val_loader = val
                    dice, hd95, asd = self.validate_volume()
                    dice_avg+=dice
                    hd95_avg+=hd95
                    asd_avg+=asd
                dice_avg /= len(self.val_loaders)
                hd95_avg /= len(self.val_loaders)
                asd_avg /= len(self.val_loaders)
                print("           avg dice:{:<13} avg hd95:{:<13} avg asd:{}".format(np.round(dice_avg,4), np.round(hd95_avg,4), np.round(asd_avg,4)))

                self.save_epoch(dice_avg, hd95_avg, asd_avg)

            stop_time = timeit.default_timer()
            spend_time = stop_time-start_time
            rest_time = spend_time * (self.max_epoch-self.epoch) // 60

            print("           Epoch time: {:<10} Rest time: {} min".format(np.round(spend_time,1), rest_time))

        self.writer.close()
