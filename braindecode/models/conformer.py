# Authors: Yonghao Song <eeyhsong@gmail.com>
#
# License: BSD (3-clause)

import os
import argparse
import numpy as np
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class Conformer(nn.Sequential):
    """

    # EEG Conformer (Conformer or CF)
    Convolutional Transformer for EEG decoding

    - Original code: https://github.com/eeyhsong/EEG-Conformer
    - Paper: https://ieeexplore.ieee.org/document/9991178

    
    ## Note 
    - Recommend to use augment the data before use Conformer, e.g. S&R in the end of the code.
    - Please refer to the original paper and code for more details.


    ## Input 
    EEG signals of shape (batch_size, 1, n_channels, n_times) - four dimensions

    
    ## Output
    (embeds, out)
    - embeds: embedding after CNN and Transformer module
    - out: classification result -> CrossEntropyLoss

    
    ## Parameters
    ### Convolution
    - kernel: kernel number of the temporal convolution layer (first layer)
    - kernel_temp_conv: kernel size of the temporal convolution layer
    - eeg_channel: number of EEG channels (kernal size of the spatial convolution layer)
    - kernel_avg_pool: kernel size of the average pooling layer
    - stride_avg_pool: stride of the average pooling layer
    - conv_drop: dropout of the convolutional layer

    ### Transformer
    - att_depth: number of self-attention layers
    - att_heads: number of attention heads
    - att_drop: dropout of the self-attention layer

    ### Classification
    - fc_dim: dimension of the fully connected layer
    - n_classes: number of classes

    """
    def __init__(self, 
                 kernel=40,
                 kernel_temp_conv=25,
                 eeg_channel=22, 
                 kernel_avg_pool=75,
                 stride_avg_pool=15,
                 conv_drop=0.5,
                 att_depth=6, 
                 att_heads=10,
                 att_drop=0.5,
                 fc_dim=2440,
                 n_classes=4, 
                 **kwargs):
        super().__init__(

            PatchEmbedding(kernel, kernel_temp_conv, eeg_channel, kernel_avg_pool, stride_avg_pool, conv_drop),
            TransformerEncoder(att_depth, kernel, att_heads, att_drop),
            ClassificationHead(kernel, fc_dim, n_classes)
        )

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, kernel, kernel_temp_conv, eeg_channel, kernel_avg_pool, stride_avg_pool, conv_drop):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, kernel, (1, kernel_temp_conv), (1, 1)),
            nn.Conv2d(kernel, kernel, (eeg_channel, 1), (1, 1)),
            nn.BatchNorm2d(kernel),
            nn.ELU(),
            nn.AvgPool2d((1, kernel_avg_pool), (1, stride_avg_pool)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(conv_drop),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(kernel, kernel, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        # x = x.unsqueeze(dim=1)
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 att_heads,
                 att_drop,
                 forward_expansion=4):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, att_heads, att_drop),
                nn.Dropout(att_drop)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=att_drop),
                nn.Dropout(att_drop)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, att_depth, emb_size, att_heads, att_drop):
        super().__init__(*[TransformerEncoderBlock(emb_size, att_heads, att_drop) for _ in range(att_depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, fc_dim, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


# # Segmentation and Reconstruction (S&R) data augmentation
# def interaug(self, timg, label, channels, samples):  
#     aug_data = []
#     aug_label = []
#     for cls4aug in range(4): # 4 is the number of classes
#         cls_idx = np.where(label == cls4aug + 1)
#         tmp_data = timg[cls_idx]
#         tmp_label = label[cls_idx]

#         tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, channels, samples))
#         # segment the original signal into 8 segments (1000/8=125) along the time dimension and randomly select one segment for each sample
#         for ri in range(int(self.batch_size / 4)):
#             for rj in range(8):
#                 rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
#                 tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
#                                                                     rj * 125:(rj + 1) * 125]
                
#         aug_data.append(tmp_aug_data)
#         aug_label.append(tmp_label[:int(self.batch_size / 4)])
#     aug_data = np.concatenate(aug_data)
#     aug_label = np.concatenate(aug_label)

#     # shuffle the augment data
#     aug_shuffle = np.random.permutation(len(aug_data))
#     aug_data = aug_data[aug_shuffle, :, :]
#     aug_label = aug_label[aug_shuffle]

#     aug_data = torch.from_numpy(aug_data).cuda()
#     aug_data = aug_data.float()
#     aug_label = torch.from_numpy(aug_label-1).cuda()
#     aug_label = aug_label.long()
#     return aug_data, aug_label
