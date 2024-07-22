import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.MetaEmbedding import MetaEmbedding


class Decoder(nn.Module):
    def __init__(self, num_classes, num_domain, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet101' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'resnet50':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn = BatchNorm(256)
        self.relu = nn.ReLU()
        # self.embedding = MetaEmbedding(304, num_domain)
        self.decoder = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       )
        self.last_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False)
        self._init_weight()

    def forward(self, x):
        feat = self.decoder(x)
        x = self.last_conv(feat)
        return x, feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, num_domain, backbone, BatchNorm):
    return Decoder(num_classes, num_domain, backbone, BatchNorm)
