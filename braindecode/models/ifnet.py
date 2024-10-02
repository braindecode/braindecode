# --------------------------------------------------------
# IFNet
# Written by Jiaheng Wang
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super(LogPowerLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x**2, dim=self.dim), 1e-4, 1e4))
        # return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=False), 1e-4, 1e4))


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Stem(nn.Module):
    def __init__(
        self, in_planes, out_planes=64, kernel_size=63, patch_size=125, radix=2
    ):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.patch_size = patch_size

        self.sconv = Conv(
            nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups=radix),
            bn=nn.BatchNorm1d(self.mid_planes),
            activation=None,
        )

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(
                Conv(
                    nn.Conv1d(
                        self.out_planes,
                        self.out_planes,
                        kernel_size,
                        1,
                        groups=self.out_planes,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    bn=nn.BatchNorm1d(self.out_planes),
                    activation=None,
                )
            )
            kernel_size //= 2

        self.interFre = InterFre()

        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)

        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]

        out = self.interFre(out)

        out = out.reshape(N, self.out_planes, T // self.patch_size, self.patch_size)
        out = self.power(out)
        out = self.dp(out)
        return out


class IFNet(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        radix,
        patch_size,
        time_points,
        num_classes,
    ):
        r"""Interactive Frequency Convolutional Neural Network V2

        :param in_planes: Number of input EEG channels
        :param out_planes: Number of output feature dimensions
        :param kernel_size: Temporal convolution kernel size
        :param radix:   Number of input frequency bands
        :param patch_size: Temporal pooling size
        :param time_points: Input window length
        :param num_classes: Number of classes
        """
        nn.Module.__init__(self)
        self.in_planes = in_planes * radix
        self.out_planes = out_planes
        self.stem = Stem(
            self.in_planes,
            self.out_planes,
            kernel_size,
            patch_size=patch_size,
            radix=radix,
        )

        self.final_layer = nn.Sequential(
            LinearWithConstraint(
                out_planes * (time_points // patch_size), num_classes, doWeightNorm=True
            ),
        )
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.stem(x)
        out = self.final_layer(out.flatten(1))
        return out
