# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)
import pytest
import torch

from braindecode.models.signal_jepa import (
    _ConvFeatureEncoder,
    _pos_encode_contineous,
    _pos_encode_time,
    _PosEncoder,
)


class TestPosEncodeTimeFunction:
    def test_shape(self):
        pos_encoding = _pos_encode_time(10, 6, 20)
        assert pos_encoding.shape == (10, 6)

    # def test_plot(self):
    #     import matplotlib.pyplot as plt
    #     n_times = 100
    #     n_dim = 100
    #     max_n_times = 100
    #     pos_encoding = _pos_encode_time(n_times, n_dim, max_n_times)
    #     plt.imshow(pos_encoding.T, aspect=n_times / n_dim)
    #     plt.ylabel('embedding dimension')
    #     plt.xlabel('time')
    #     plt.colorbar()
    #     plt.show()


class TestPosEncodeContineousFunction:
    def test_shape(self):
        pos_encoding = _pos_encode_contineous(0.5, 0, 1, 6)
        assert pos_encoding.shape == (6,)

    def test_sanity(self):
        torch.testing.assert_close(
            _pos_encode_contineous(0.75, 0, 1, 6),
            _pos_encode_contineous(0.5, -1, 1, 6),
        )
        torch.testing.assert_close(
            _pos_encode_contineous(0.75, 0, 1, 6),
            _pos_encode_contineous(-0.25, -1, 0, 6),
        )

    # def test_plot(self):
    #     import matplotlib.pyplot as plt
    #     n_dim = 100
    #     x0 = -.001
    #     x1 = .001
    #     steps = 100
    #     x = torch.linspace(x0, x1, steps)
    #     out = torch.empty((len(x), n_dim))
    #     for i, xx in enumerate(x):
    #         out[i] = _pos_encode_contineous(xx, 0, 10 * x1, n_dim)
    #     plt.imshow(out.T, aspect=len(x) / n_dim)
    #     plt.ylabel('embedding dimension')
    #     plt.xlabel('x')
    #     plt.colorbar()
    #     plt.show()


class TestConvFeatureEncoderModule:
    def test_forward__n_times_out(self):
        model = _ConvFeatureEncoder([(32, 10, 5), (64, 5, 2), (128, 5, 2)], channels=3)
        x = torch.rand(2, 3, 103)
        y = model(x)
        n_times_out = model.n_times_out(103)
        assert y.shape == (2, 3 * n_times_out, 128)


class TestPosEncoderModule:
    @pytest.fixture(scope="function")
    def batch(self):
        return dict(
            local_features=torch.zeros(2, 4 * 10, 100),
            ch_idxs=torch.tensor([[1, 2, 0, 0]] * 2),
        )

    @pytest.fixture(scope="function")
    def model(self):
        return _PosEncoder(
            spat_dim=40,
            time_dim=60,
            ch_locs=[[1.0, 2.0], [0.0, 1.0], [0.0, 0.0]],
            sfreq_features=128,
        )

    def test_forward(self, batch, model):
        y = model(**batch)
        assert y.shape == (2, 4 * 10, 100)
        y = y.view(2, 4, 10, 100)
        assert (y[:, 2, :, :40] == y[:, 3, :, :40]).all()  # same unknown embedding
        assert not (y[:, 0, :, :40] == y[:, 1, :, :40]).all()  # different embeddings
        assert not (y[:, 0, :, :40] == y[:, 2, :, :40]).all()  # different embeddings
        # same spat embedding for all time samples:
        assert all((y[:, :, 0, :40] == y[:, :, i, :40]).all() for i in range(1, 10))
        # same time embedding for all channels and all batch elements:
        assert all(
            (y[0, 0, :, 60:] == y[i, j, :, 60:]).all()
            for j in range(1, 4)
            for i in range(2)
        )

    def test_forward_forward(self, batch, model):
        _ = model(**batch)
        batch["local_features"] = batch["local_features"].tile(1, 2, 1)
        _ = model(**batch)
