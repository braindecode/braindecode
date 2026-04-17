# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import pytest
import torch

from braindecode.models.signal_jepa import (
    _PRETRAIN_CHS_INFO,
    SignalJEPA,
    SignalJEPA_Contextual,
    _ConvFeatureEncoder,
    _pos_encode_contineous,
    _pos_encode_time,
    _PosEncoder,
    _resolve_channel_embedding_config,
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
            channel_locations=[[1.0, 2.0], [0.0, 1.0], [0.0, 0.0]],
            ch_idxs=torch.tensor([0, 1, 2, 0], dtype=torch.long),
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


class TestResolveChannelEmbeddingConfig:
    @staticmethod
    def _fake_user_chs(names):
        # Minimal chs_info-like list. Locs are arbitrary but present.
        return [{"ch_name": n, "loc": [0.1, 0.2, 0.3]} for n in names]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="channel_embedding must be"):
            _resolve_channel_embedding_config("bogus", None)

    def test_scratch_without_chs_info_raises(self):
        with pytest.raises(ValueError, match="chs_info is required"):
            _resolve_channel_embedding_config("scratch", None)

    def test_scratch_with_chs_info(self):
        user = self._fake_user_chs(["A", "B", "C"])
        eff, locs, idxs = _resolve_channel_embedding_config("scratch", user)
        assert eff == user
        assert locs == [ch["loc"] for ch in user]
        assert torch.equal(idxs, torch.arange(3, dtype=torch.long))

    def test_pretrain_aligned_without_chs_info(self):
        eff, locs, idxs = _resolve_channel_embedding_config(
            "pretrain_aligned", None
        )
        assert eff == _PRETRAIN_CHS_INFO
        assert locs == [ch["loc"] for ch in _PRETRAIN_CHS_INFO]
        assert torch.equal(idxs, torch.arange(62, dtype=torch.long))

    def test_pretrain_aligned_with_subset(self):
        # Pick three names from the pretrain set in non-monotonic order.
        pretrain_names = [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]
        chosen = [pretrain_names[5], pretrain_names[0], pretrain_names[10]]
        user = self._fake_user_chs(chosen)
        eff, locs, idxs = _resolve_channel_embedding_config(
            "pretrain_aligned", user
        )
        assert eff == user  # user chs_info preserved
        assert locs == [ch["loc"] for ch in _PRETRAIN_CHS_INFO]  # 62-long
        assert torch.equal(idxs, torch.tensor([5, 0, 10], dtype=torch.long))

    def test_pretrain_aligned_case_insensitive(self):
        # Match regardless of casing on either side.
        pretrain_names = [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]
        real = pretrain_names[0]
        flipped = real.swapcase()
        user = self._fake_user_chs([flipped])
        eff, locs, idxs = _resolve_channel_embedding_config(
            "pretrain_aligned", user
        )
        assert idxs.tolist() == [0]

    def test_pretrain_aligned_channel_outside_raises(self):
        user = self._fake_user_chs(["__not_a_real_ch__"])
        with pytest.raises(ValueError) as exc:
            _resolve_channel_embedding_config("pretrain_aligned", user)
        msg = str(exc.value)
        assert "__not_a_real_ch__" in msg
        assert "channel_embedding='scratch'" in msg
        assert "signal-jepa_without-chans" in msg

    def test_returns_long_tensor(self):
        user = self._fake_user_chs(["A", "B"])
        _, _, idxs = _resolve_channel_embedding_config("scratch", user)
        assert idxs.dtype == torch.long


class TestPosEncoderBuffer:
    def test_default_ch_idxs_is_non_persistent(self):
        pe = _PosEncoder(
            spat_dim=4,
            time_dim=6,
            channel_locations=[[0.0, 0.0], [1.0, 1.0]],
            ch_idxs=torch.tensor([0, 1], dtype=torch.long),
            sfreq_features=1.0,
        )
        # Buffer is registered and accessible.
        assert torch.equal(pe.default_ch_idxs, torch.tensor([0, 1]))
        # Buffer is NOT in state_dict (persistent=False).
        assert "default_ch_idxs" not in pe.state_dict()

    def test_default_ch_idxs_used_when_forward_kwarg_is_none(self):
        pe = _PosEncoder(
            spat_dim=4,
            time_dim=6,
            channel_locations=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            ch_idxs=torch.tensor([2, 0], dtype=torch.long),
            sfreq_features=1.0,
        )
        # 2 user channels, each gets 5 time steps of features.
        local_features = torch.zeros(1, 2 * 5, 10)
        out_default = pe(local_features)
        out_explicit = pe(
            local_features,
            ch_idxs=torch.tensor([[2, 0]], dtype=torch.long),
        )
        torch.testing.assert_close(out_default, out_explicit)

    def test_default_ch_idxs_moves_with_to_device(self):
        pe = _PosEncoder(
            spat_dim=4,
            time_dim=6,
            channel_locations=[[0.0, 0.0], [1.0, 1.0]],
            ch_idxs=torch.tensor([0, 1], dtype=torch.long),
            sfreq_features=1.0,
        )
        # .to() is a no-op for CPU but still exercises the buffer registration.
        pe = pe.to(torch.device("cpu"))
        assert pe.default_ch_idxs.device == torch.device("cpu")


class TestSignalJEPAChannelEmbedding:
    @staticmethod
    def _user_chs(names):
        # Use realistic electrode coordinates (typical 10-20 system positions)
        return [{"ch_name": n, "loc": [0.1 * i, 0.2 * i, 0.3 * i]} for i, n in enumerate(names)]

    def test_scratch_default_is_scratch_mode(self):
        user = self._user_chs(["A", "B", "C"])
        m = SignalJEPA(chs_info=user, n_times=128, sfreq=128)
        # Embedding table has N=3 rows in scratch mode.
        assert m.pos_encoder.pos_encoder_spat.weight.shape[0] == 3
        assert m._channel_embedding == "scratch"

    def test_pretrain_aligned_no_chs_info(self):
        m = SignalJEPA(
            chs_info=None,
            n_times=128,
            sfreq=128,
            channel_embedding="pretrain_aligned",
        )
        assert m.n_chans == 62
        # Table has 62 rows.
        assert m.pos_encoder.pos_encoder_spat.weight.shape[0] == 62

    def test_pretrain_aligned_with_subset(self):
        pretrain_names = [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]
        chosen = [pretrain_names[5], pretrain_names[0], pretrain_names[10]]
        user = self._user_chs(chosen)
        m = SignalJEPA(
            chs_info=user,
            n_times=128,
            sfreq=128,
            channel_embedding="pretrain_aligned",
        )
        # Table has 62 rows regardless of user count.
        assert m.pos_encoder.pos_encoder_spat.weight.shape[0] == 62
        # User count is N=3.
        assert m.n_chans == 3
        # Mapping points to the right rows.
        assert torch.equal(
            m.pos_encoder.default_ch_idxs, torch.tensor([5, 0, 10])
        )

    def test_pretrain_aligned_bad_channel_raises(self):
        user = self._user_chs(["__NOT_A_REAL_CHANNEL__"])
        with pytest.raises(ValueError, match="not in the Signal-JEPA"):
            SignalJEPA(
                chs_info=user,
                n_times=128,
                sfreq=128,
                channel_embedding="pretrain_aligned",
            )

    def test_scratch_without_chs_info_raises(self):
        with pytest.raises(ValueError, match="chs_info is required"):
            SignalJEPA(n_chans=3, n_times=128, sfreq=128)


class TestSignalJEPAForwardAPIChange:
    def test_forward_rejects_ch_idxs_kwarg(self):
        m = SignalJEPA(
            chs_info=[
                {"ch_name": "A", "loc": [0.1, 0.2, 0.3]},
                {"ch_name": "B", "loc": [0.4, 0.5, 0.6]},
            ],
            n_times=2000,
            sfreq=128,
        )
        X = torch.randn(1, 2, 2000)
        with pytest.raises(TypeError):
            m(X, ch_idxs=torch.tensor([[0, 1]]))

    def test_contextual_forward_rejects_ch_idxs_kwarg(self):
        m = SignalJEPA_Contextual(
            chs_info=[
                {"ch_name": "A", "loc": [0.1, 0.2, 0.3]},
                {"ch_name": "B", "loc": [0.4, 0.5, 0.6]},
            ],
            n_times=2000,
            sfreq=128,
            n_outputs=4,
        )
        X = torch.randn(1, 2, 2000)
        with pytest.raises(TypeError):
            m(X, ch_idxs=torch.tensor([[0, 1]]))

    def test_forward_runs_in_pretrain_aligned_mode(self):
        pretrain_names = [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]
        chosen = pretrain_names[:3]
        user = [
            {"ch_name": n, "loc": [0.1 * i, 0.2 * i, 0.3 * i]}
            for i, n in enumerate(chosen)
        ]
        m = SignalJEPA(
            chs_info=user,
            n_times=2000,
            sfreq=128,
            channel_embedding="pretrain_aligned",
        )
        X = torch.randn(2, 3, 2000)
        y = m(X)
        # Output is (batch, n_chans * n_times_out, emb_dim). We just check
        # that it runs and has the right batch dim.
        assert y.shape[0] == 2

    def test_contextual_channel_embedding_forwarded(self):
        pretrain_names = [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]
        user = [
            {"ch_name": n, "loc": [0.1 * i, 0.2 * i, 0.3 * i]}
            for i, n in enumerate(pretrain_names[:3])
        ]
        m = SignalJEPA_Contextual(
            chs_info=user,
            n_times=2000,
            sfreq=128,
            n_outputs=4,
            channel_embedding="pretrain_aligned",
        )
        assert m._channel_embedding == "pretrain_aligned"
        assert m.pos_encoder.pos_encoder_spat.weight.shape[0] == 62


class TestContextualFromPretrainedTransfer:
    def test_from_pretrained_transfer_with_chs_info(self):
        # Build a source SignalJEPA with varying per-channel locs so that
        # _ChannelEmbedding.reset_parameters does not divide by zero.
        src_user = [
            {"ch_name": n, "loc": [0.01 * (i + 1), 0.02 * (i + 1), 0.03 * (i + 1)]}
            for i, n in enumerate(("A", "B", "C"))
        ]
        src = SignalJEPA(chs_info=src_user, n_times=128, sfreq=128)

        # Transfer into a downstream classifier with chs_info set.
        # Historically this raised AttributeError via the dead
        # set_fixed_ch_names call; after Task 6 it should just work.
        dst_user = [
            {"ch_name": n, "loc": [0.01 * (i + 1), 0.02 * (i + 1), 0.03 * (i + 1)]}
            for i, n in enumerate(("A", "B", "C"))
        ]
        new_model = SignalJEPA_Contextual.from_pretrained(
            src,
            n_outputs=4,
            chs_info=dst_user,
        )
        assert new_model.n_outputs == 4

    def test_from_pretrained_transfer_pretrain_aligned_different_subset(self):
        # Source uses pretrain_aligned with the first 3 channels of the
        # pretraining set. Downstream uses a different 3-channel subset.
        # The transferred default_ch_idxs MUST reflect the downstream
        # channels, not the source channels.
        pretrain_names = [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]
        src_user = [
            {
                "ch_name": n,
                "loc": [0.01 * (i + 1), 0.02 * (i + 1), 0.03 * (i + 1)],
            }
            for i, n in enumerate(pretrain_names[:3])  # rows 0, 1, 2
        ]
        src = SignalJEPA(
            chs_info=src_user,
            n_times=128,
            sfreq=128,
            channel_embedding="pretrain_aligned",
        )
        assert torch.equal(
            src.pos_encoder.default_ch_idxs, torch.tensor([0, 1, 2])
        )

        dst_names = [pretrain_names[5], pretrain_names[0], pretrain_names[10]]
        dst_user = [
            {
                "ch_name": n,
                "loc": [0.01 * (i + 1), 0.02 * (i + 1), 0.03 * (i + 1)],
            }
            for i, n in enumerate(dst_names)
        ]
        new_model = SignalJEPA_Contextual.from_pretrained(
            src,
            n_outputs=4,
            chs_info=dst_user,
        )
        # The buffer must be recomputed for the destination channels.
        assert torch.equal(
            new_model.pos_encoder.default_ch_idxs, torch.tensor([5, 0, 10])
        )
        # The 62-row embedding table is preserved unchanged.
        assert new_model.pos_encoder.pos_encoder_spat.weight.shape[0] == 62
        torch.testing.assert_close(
            new_model.pos_encoder.pos_encoder_spat.weight,
            src.pos_encoder.pos_encoder_spat.weight,
        )
        # The channel_embedding mode is preserved on the new model.
        assert new_model._channel_embedding == "pretrain_aligned"


try:
    import huggingface_hub  # noqa: F401

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


@pytest.mark.skipif(not HAS_HF_HUB, reason="requires huggingface_hub")
class TestHFRoundTrip:
    def test_channel_embedding_survives_save_load(self, tmp_path):
        pretrain_names = [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]
        user = [
            {
                "ch_name": n,
                "loc": [0.01 * (i + 1), 0.02 * (i + 1), 0.03 * (i + 1)],
            }
            for i, n in enumerate(pretrain_names[:3])
        ]
        model = SignalJEPA(
            chs_info=user,
            n_times=128,
            sfreq=128,
            channel_embedding="pretrain_aligned",
        )

        model.save_pretrained(str(tmp_path))
        reloaded = SignalJEPA.from_pretrained(str(tmp_path))

        assert reloaded._channel_embedding == "pretrain_aligned"
        assert reloaded.pos_encoder.pos_encoder_spat.weight.shape[0] == 62
        # The non-persistent buffer must NOT be in the saved state_dict.
        assert "pos_encoder.default_ch_idxs" not in reloaded.state_dict()

    def test_user_override_chs_info_at_load_time(self, tmp_path):
        # Save with the full 62 channels, load with a 3-channel override.
        model = SignalJEPA(
            chs_info=None,  # triggers _PRETRAIN_CHS_INFO
            n_times=128,
            sfreq=128,
            channel_embedding="pretrain_aligned",
        )

        pretrain_names = [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]
        override_user = [
            {
                "ch_name": n,
                "loc": [0.01 * (i + 1), 0.02 * (i + 1), 0.03 * (i + 1)],
            }
            for i, n in enumerate(pretrain_names[:3])
        ]

        model.save_pretrained(str(tmp_path))
        reloaded = SignalJEPA.from_pretrained(
            str(tmp_path), chs_info=override_user
        )

        assert reloaded.n_chans == 3
        # Embedding table keeps 62 rows.
        assert reloaded.pos_encoder.pos_encoder_spat.weight.shape[0] == 62
        # Weights loaded correctly (not reinitialized): compare to source.
        torch.testing.assert_close(
            reloaded.pos_encoder.pos_encoder_spat.weight,
            model.pos_encoder.pos_encoder_spat.weight,
        )
        # Mapping points to rows 0, 1, 2 of the pretrain set.
        assert torch.equal(
            reloaded.pos_encoder.default_ch_idxs, torch.tensor([0, 1, 2])
        )
