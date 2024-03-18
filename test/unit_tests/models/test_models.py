# Authors: Alexandre Gramfort
#          Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#          Daniel Wilson <dan.c.wil@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com
#
# License: BSD-3

from functools import partial

from collections import OrderedDict
from sklearn.utils import check_random_state
from torch import nn

import numpy as np
import torch
import pytest

from braindecode.models import (
    Labram,
)


@pytest.fixture
def default_labram_params():
    return {
        'n_times': 1000,
        'n_chans': 64,
        'patch_size': 200,
        'sfreq': 200,
        'qk_norm': partial(nn.LayerNorm, eps=1e-6),
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'mlp_ratio': 4,
        'n_outputs': 2,
    }


def test_model_trainable_parameters_labram(default_labram_params):
    """
    Test the number of trainable parameters in Labram model based on the
    paper values.

    Parameters
    ----------
    default_labram_params: dict with default parameters for Labram model

    """
    labram_base = Labram(n_layers=12, att_num_heads=12,
                         **default_labram_params)

    labram_base_parameters = (labram_base.get_torchinfo_statistics()
                              .trainable_params)

    # We added some parameters layers in the segmentation step to match the
    # braindecode convention.
    assert labram_base_parameters == 5789746  # ~ 5.8 M matching the paper

    labram_large = Labram(n_layers=24, att_num_heads=16, out_channels=16,
                          emb_size=400, **default_labram_params)
    labram_large_parameters = (labram_large.get_torchinfo_statistics()
                               .trainable_params)

    assert labram_large_parameters == 46262322  # ~ 46 M matching the paper

    labram_huge = Labram(n_layers=48, att_num_heads=16, out_channels=32,
                         emb_size=800, **default_labram_params)

    labram_huge_parameters = (labram_huge.get_torchinfo_statistics()
                              .trainable_params)

    assert labram_huge_parameters == 369141514  # 369M


def test_labram_returns(default_labram_params):
    """
    Testing if the model is returning the correct shapes for the different
    return options.

    Parameters
    ----------
    default_labram_params: dict with default parameters for Labram model

    """
    labram_base = Labram(n_layers=12, att_num_heads=12,
                         **default_labram_params)
    # Defining a random data
    X = torch.rand(1, 32, 1000)

    with torch.no_grad():
        out = labram_base(X, return_all_tokens=False,
                          return_patch_tokens=False)

        assert out.shape == torch.Size([1, 200])

        out_patches = labram_base(X, return_all_tokens=False,
                                  return_patch_tokens=True)

        assert out_patches.shape == torch.Size([1, 160, 200])

        out_all_tokens = labram_base(X, return_all_tokens=True,
                                     return_patch_tokens=False)
        assert out_all_tokens.shape == torch.Size([1, 161, 200])


def test_labram_without_pos_embed(default_labram_params):
    labram_base_not_pos_emb = Labram(n_layers=12, att_num_heads=12,
                                     use_abs_pos_emb=False,
                                     **default_labram_params)

    X = torch.rand(1, 32, 1000)

    with torch.no_grad():
        out_without_pos_emb = labram_base_not_pos_emb(X)
        assert out_without_pos_emb.shape == torch.Size([1, 2])
