import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.aspp import build_aspp
from networks.decoder import build_decoder
from networks.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                 sync_bn=False, num_domains=3, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        # self.bag_num = bag_num
        # self.centroids = nn.Parameter(torch.randn(num_domain, 304, 64, 64), requires_grad=False)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, num_domains, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, mask):
        x_ = self.backbone(input, mask)
        x, _ = self.aspp(x_)
        x, feat = self.decoder(x)
        pred = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return pred, feat

    # def domain_classifier(self):


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_para(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        # for param in self.aspp.parameters():
        #     param.requires_grad = False

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


# if __name__ == "__main__":
#     model = DeepLab_DomainCode(backbone='mobilenet', output_stride=16)
#     model.eval()
#     input = torch.rand(1, 3, 513, 513)
#     output = model(input)
#     print(output.size())