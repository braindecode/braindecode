# Authors: Yue Wang <yuewangx@mit.edu>
#
# Code adapted from https://github.com/WangYueFt/dgcnn
#
# License: BSD (3-clause)

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin


def knn(x, k):
    # x: (B, n_times, n_chans)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, n_chans, n_chans)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)       # (B, 1, n_chans)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, n_chans, n_chans)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]      # (B, n_chans, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0) # B
    n_chans = x.size(2) # n_chans
    x = x.view(batch_size, -1, n_chans) 
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, n_chans, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*n_chans

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*n_chans, -1)[idx, :]
    feature = feature.view(batch_size, n_chans, k, num_dims) 
    x = x.view(batch_size, n_chans, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class DGCNN(EEGModuleMixin, nn.Module):
    def __init__(self,
            n_outputs=40,
            n_chans=None,
            chs_info=None,
            n_times=None,
            input_window_seconds=None,
            sfreq=None,
            emb_dims=1024,
            k=20,
            dropout=0.5):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds
        )

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(self.n_times * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.final_layer = nn.Linear(256, self.n_outputs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size=x.size(0)
        x = x.transpose(1, 2)  # (B, n_chans, n_times) → (B, n_times, n_chans)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.final_layer(x)
        return x