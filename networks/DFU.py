import torch
import torch.nn as nn
import numpy as np

class DFU(nn.Module):
    """
        domain feature unifying module (DFU)
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, lambda_=0.1, eps=1e-6):
        super(DFU, self).__init__()
        self.eps = eps
        self.p = p
        self.lambda_ = lambda_
        self.factor = 1.0
        self.bag_mean = None
        self.bag_std = None
        # self.update_rate = 0.1

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    # def sqrtvar(self, x):
    #     B = x[-1].shape[0]
    #     x = torch.cat(x, dim=0)
    #     t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
    #     t = t.repeat(B, 1)
    #     return t
    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if not self.training:
            return x

        if len(x.shape) == 3:
            mean = x.mean(dim=[1], keepdim=False)
            std = (x.var(dim=[1], keepdim=False) + self.eps).sqrt()
        else:
            mean = x.mean(dim=[2, 3], keepdim=False)
            std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        if self.bag_mean is None:
            self.bag_mean = mean.detach()
            self.bag_std = std.detach()
        else:
            if self.bag_mean.shape == mean.shape:
                self.bag_mean  = self.lambda_ * self.bag_mean + (1-self.lambda_) * mean.detach()
                self.bag_std = self.lambda_ * self.bag_std + (1 - self.lambda_) * std.detach()
            else:
                self.bag_mean = mean.detach()
                self.bag_std = std.detach()
            # else:
            #     B, C = mean.shape
            #     self.bag_mean = self.update_rate * self.bag_mean[:B, :] + (1-self.update_rate) * mean.detach()
            #     self.bag_std = self.update_rate * self.bag_std[:B, :] + (1 - self.update_rate) * std.detach()

        if (not self.training) or (np.random.random()) > self.p:
            return x

        sqrtvar_mu = self.sqrtvar(self.bag_mean)
        sqrtvar_std = self.sqrtvar(self.bag_std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        if len(x.shape) == 3:
            x = (x - mean.reshape(x.shape[0], 1, x.shape[2])) / std.reshape(x.shape[0], 1, x.shape[2])
            x = x * gamma.reshape(x.shape[0], 1, x.shape[2]) + beta.reshape(x.shape[0], 1, x.shape[2])
        else:
            x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
            x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x



