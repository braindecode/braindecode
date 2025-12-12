# Authors: Jonathan Lys <jonathan.lys@imt-atlantique.org>
#
# License: BSD (3-clause)
from shutil import rmtree

import torch
from transformers import AutoModel

from braindecode.models.reve import REVE, RevePositionBank


class TestHFLoadingReve:
    batch_size = 2
    n_chans = 32
    n_times = 1000
    n_outputs = 10

    cache_dir = "./cache"
    model_id = "brain-bzh/reve-base"
    positions_id = "brain-bzh/reve-positions"

    def _init_pos_banks(self):
        """Helper to initialize both position banks"""
        pos_bank_hf = AutoModel.from_pretrained(
            self.positions_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        pos_bank_bd = RevePositionBank()

        return pos_bank_hf, pos_bank_bd

    def _init_models(self):
        """Helper to initialize both models"""
        model_hf = model_hf = AutoModel.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        model_bd: REVE = REVE.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            n_times=self.n_times,
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
        )

        return model_hf, model_bd

    def _cleanup(self):
        """Used to clean the cache directory"""
        rmtree(self.cache_dir)

    def test_positions(self):
        """Test that the positions from both implementations match"""
        pos_bank_hf, pos_bank_bd = self._init_pos_banks()

        all_pos_hf = pos_bank_hf.get_all_positions()
        all_pos_bd = pos_bank_bd.get_all_positions()

        assert all_pos_hf == all_pos_bd, "mismatch"

        for pos in all_pos_bd:
            pos_hf = pos_bank_hf([pos])
            pos_bd = pos_bank_bd([pos])

            assert torch.allclose(pos_hf, pos_bd)

        self._cleanup()

    def test_model_outputs(self):
        """Test that the outputs from both implementations match"""
        ch_list = [f"E{i + 1}" for i in range(self.n_chans)]

        torch.manual_seed(42)
        eeg_input = torch.randn(self.batch_size, self.n_chans, self.n_times)

        pos_bank_hf, _pos_bank_bd = self._init_pos_banks()
        model_hf, model_bd = self._init_models()

        pos_hf = pos_bank_hf(ch_list)
        pos_hf = pos_hf.unsqueeze(0).repeat(self.batch_size, 1, 1)

        pos_bd = model_bd.get_positions(ch_list)
        pos_bd = pos_bd.unsqueeze(0).repeat(self.batch_size, 1, 1)

        assert torch.allclose(pos_hf, pos_bd)

        # return_output is True to bypass the last layer
        output_bd = model_bd(eeg_input, pos_bd, return_output=True)[-1]
        output_hf = model_hf(eeg_input, pos_hf, return_output=True)[-1]

        assert torch.allclose(output_hf, output_bd)

        self._cleanup()
