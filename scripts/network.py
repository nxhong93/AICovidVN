import torch
import torch.nn as nn
from torch.nn import functional as F


class MfccNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(MfccNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3,
                                            padding=1, stride=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(0.2),
                                  nn.Conv2d(64, 128, kernel_size=3,
                                            padding=1, stride=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(0.2),
                                  nn.Conv2d(128, 128, kernel_size=3,
                                            padding=1, stride=1),
                                  nn.BatchNorm2d(128))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(128, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(64, out_channels))

    def forward(self, x):
        out = self.conv(x)
        out1 = F.adaptive_avg_pool2d(out, 1)
        out2 = F.adaptive_max_pool2d(out, 1)
        out = out1 + out2
        out = self.fc(out)

        return out


def gem(x: torch.Tensor, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
