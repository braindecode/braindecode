"""Distributed EEGCLIP regression: global negatives and DDP gradient parity."""

import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from braindecode.models import EEGCLIP


def _worker(rank, sizes, rendezvous, result_path):
    # Deterministic CPU/Gloo demonstration of the *actual model method*.
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).resolve().as_uri(),
        timeout=timedelta(seconds=40),
        rank=rank, world_size=len(sizes),
    )
    try:
        torch.manual_seed(111)
        n = sum(sizes)
        eeg = torch.randn(n, 4, dtype=torch.float64)
        text = torch.randn(n, 4, dtype=torch.float64)
        projection_init = torch.randn(4, 4, dtype=torch.float64) * 0.2
        model = EEGCLIP(
            n_chans=4, n_times=20, n_outputs=4,
            eeg_encoder=nn.Identity(), eeg_embedding_dim=4,
            text_embedding_dim=4, projection_layers=1,
        ).double()
        projection = projection_init.clone().requires_grad_()
        start = sum(sizes[:rank])
        stop = start + sizes[rank]
        local_eeg = eeg[start:stop] @ projection
        local_text = text[start:stop] @ projection

        distributed_loss = model.contrastive_loss(
            local_eeg, local_text, distributed=True
        )
        distributed_loss.backward()
        # Emulate DDP's default parameter-gradient mean over all ranks.
        projection_grad = projection.grad.detach().clone()
        scale_grad = model.logit_scale.grad.detach().clone()
        dist.all_reduce(projection_grad)
        dist.all_reduce(scale_grad)
        projection_grad /= len(sizes)
        scale_grad /= len(sizes)

        ref_projection = projection_init.clone().requires_grad_()
        ref_scale = model.logit_scale.detach().clone().requires_grad_()
        ref_eeg = eeg @ ref_projection
        ref_text = text @ ref_projection
        logits = ref_scale * (ref_eeg @ ref_text.T)
        indices = torch.arange(n)
        central_loss = (
            F.cross_entropy(logits, indices)
            + F.cross_entropy(logits.T, indices)
        ) / 2
        central_loss.backward()
        errors = torch.stack([
            (projection_grad - ref_projection.grad).abs().max(),
            (scale_grad - ref_scale.grad).abs(),
        ])
        rank_loss = distributed_loss.detach().clone()
        dist.all_reduce(rank_loss)
        rank_loss /= len(sizes)
        errors = torch.cat([
            errors.flatten(), (rank_loss - central_loss).abs().reshape(1)
        ])
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if rank == 0:
            with open(result_path, "w", encoding="utf-8") as out:
                json.dump({"max_abs_error": float(errors.max())}, out)
    finally:
        dist.destroy_process_group()


def _empty_rank_worker(rank, rendezvous):
    # Error across ALL ranks, not a rank-local exception before a collective.
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).resolve().as_uri(),
        timeout=timedelta(seconds=40),
        rank=rank, world_size=2,
    )
    try:
        model = EEGCLIP(
            n_chans=4, n_times=20, n_outputs=4,
            eeg_encoder=nn.Identity(), eeg_embedding_dim=4,
            text_embedding_dim=4, projection_layers=1,
        )
        local = torch.empty(rank, 4)
        with pytest.raises(ValueError, match="Every rank needs"):
            model.contrastive_loss(local, local, distributed=True)
        # A successful barrier proves both ranks exited the collective.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Gloo file rendezvous is Linux/macOS only"
)
def test_eegclip_distributed_rejects_empty_rank_without_deadlock():
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo distributed backend is unavailable")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _empty_rank_worker,
            args=(os.path.join(directory, "empty_rank"),),
            nprocs=2,
            join=True,
        )


def test_eegclip_distributed_falls_back_to_local_without_process_group():
    model = EEGCLIP(
        n_chans=4, n_times=20, n_outputs=4,
        eeg_encoder=nn.Identity(), eeg_embedding_dim=4,
        text_embedding_dim=4, projection_layers=1,
    )
    eeg, text = torch.randn(3, 4), torch.randn(3, 4)
    expected = model.contrastive_loss(eeg, text)
    actual = model.contrastive_loss(eeg, text, distributed=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("sizes", [(2, 3), (1, 5), (3, 2)])
@pytest.mark.skipif(
    sys.platform == "win32", reason="Gloo file rendezvous is Linux/macOS only"
)
def test_eegclip_distributed_matches_global_loss_and_ddp_gradients(sizes):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo distributed backend is unavailable")
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "gloo_init")
        result_path = os.path.join(directory, "result.json")
        mp.spawn(
            _worker, args=(sizes, rendezvous, result_path),
            nprocs=len(sizes), join=True,
        )
        with open(result_path, encoding="utf-8") as result:
            assert json.load(result)["max_abs_error"] < 1e-11


def test_eegclip_distributed_requires_paired_embeddings():
    model = EEGCLIP(
        n_chans=4, n_times=20, n_outputs=4,
        eeg_encoder=nn.Identity(), eeg_embedding_dim=4,
        text_embedding_dim=4, projection_layers=1,
    )
    with pytest.raises(ValueError, match="matching 2D"):
        model.contrastive_loss(torch.randn(2, 4), torch.randn(3, 4))
    with pytest.raises(ValueError, match="nonempty"):
        model.contrastive_loss(torch.empty(0, 4), torch.empty(0, 4))


def _run_with_rank_zero_in_test_process(worker, *args):
    """Execute rank zero inside pytest so coverage sees distributed branches."""
    # mp.spawn() places *every* rank in a child process, so subprocess coverage
    # may report the entire distributed branch as uncovered despite passing
    # multiprocess tests. This is a real two-rank Gloo test, not a mock.
    peer = mp.get_context("spawn").Process(target=worker, args=(1, *args))
    peer.start()
    try:
        worker(0, *args)
        peer.join(timeout=90)
        assert peer.exitcode == 0, f"Peer rank failed: {peer.exitcode}"
    finally:
        if peer.is_alive():
            peer.terminate()
            peer.join(timeout=15)


def _preflight_shape_worker(rank, rendezvous, case):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).resolve().as_uri(),
        timeout=timedelta(seconds=40),
        rank=rank,
        world_size=2,
    )
    try:
        model = EEGCLIP(
            n_chans=4,
            n_times=20,
            n_outputs=4,
            eeg_encoder=nn.Identity(),
            eeg_embedding_dim=4,
            text_embedding_dim=4,
            projection_layers=1,
        )
        dim = 3 if rank == 1 and case == "different_rank_width" else 4
        dtype = (
            torch.float64
            if rank == 1 and case == "different_rank_dtype"
            else torch.float32
        )
        eeg = torch.randn(2, dim, dtype=dtype)
        text_dim = 3 if rank == 1 and case == "mismatched_pair" else dim
        text_dtype = (
            torch.float64
            if rank == 1 and case == "mismatched_local_dtype"
            else dtype
        )
        text = torch.randn(2, text_dim, dtype=text_dtype)
        if case in ("mismatched_pair", "mismatched_local_dtype"):
            error = "matching 2D"
        elif case == "different_rank_width":
            error = "same embedding dimension"
        else:
            error = "same embedding dtype"
        with pytest.raises(ValueError, match=error):
            model.contrastive_loss(eeg, text, distributed=True)
        # A successful barrier establishes that *both* ranks left the
        # preflight collectively instead of one rank hanging in all_gather.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Autograd Gloo all-gather Windows not verified"
)
@pytest.mark.parametrize(
    "case",
    [
        "mismatched_pair",
        "different_rank_width",
        "different_rank_dtype",
        "mismatched_local_dtype",
    ],
)
def test_eegclip_distributed_preflight_rejects_malformed_rank(tmp_path, case):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo distributed backend is unavailable")
    _run_with_rank_zero_in_test_process(
        _preflight_shape_worker, str(tmp_path / f"preflight-{case}"), case
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="Autograd Gloo all-gather Windows not verified"
)
def test_eegclip_distributed_real_collectives_visible_to_coverage(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo distributed backend is unavailable")
    result_path = tmp_path / "rank-zero-covered.json"
    _run_with_rank_zero_in_test_process(
        _worker, (2, 3), str(tmp_path / "covered-gloo"), str(result_path)
    )
    with result_path.open(encoding="utf-8") as result:
        assert json.load(result)["max_abs_error"] < 1e-11


class _RealDDPContrastive(nn.Module):
    """Train the actual EEGCLIP loss through a genuine DDP parameter reducer."""

    def __init__(self):
        super().__init__()
        self.clip = EEGCLIP(
            n_chans=4,
            n_times=20,
            n_outputs=4,
            eeg_encoder=nn.Identity(),
            eeg_embedding_dim=4,
            text_embedding_dim=4,
            projection_layers=1,
        ).double()
        self.projection = nn.Linear(4, 4, bias=False, dtype=torch.float64)

    def forward(self, eeg, text, *, distributed=False):
        return self.clip.contrastive_loss(
            self.projection(eeg),
            self.projection(text),
            distributed=distributed,
        )


def _actual_ddp_worker(rank, rendezvous, result_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).resolve().as_uri(),
        timeout=timedelta(seconds=40),
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(111)
        eeg = torch.randn(5, 4, dtype=torch.float64)
        text = torch.randn(5, 4, dtype=torch.float64)
        torch.manual_seed(222)
        trained = _RealDDPContrastive()
        ddp = nn.parallel.DistributedDataParallel(
            trained, find_unused_parameters=True
        )
        start, stop = ((0, 2) if rank == 0 else (2, 5))
        loss = ddp(eeg[start:stop], text[start:stop], distributed=True)
        loss.backward()
        projection_grad = trained.projection.weight.grad.detach().clone()
        scale_grad = trained.clip.logit_scale.grad.detach().clone()

        torch.manual_seed(222)
        reference = _RealDDPContrastive()
        central_loss = reference(eeg, text)
        central_loss.backward()
        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss)
        mean_loss /= 2
        errors = torch.stack(
            [
                (projection_grad - reference.projection.weight.grad).abs().max(),
                (scale_grad - reference.clip.logit_scale.grad).abs().max(),
                (mean_loss - central_loss).abs(),
            ]
        )
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if rank == 0:
            with open(result_path, "w", encoding="utf-8") as result:
                json.dump({"max_abs_error": float(errors.max())}, result)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32", reason="DDP CPU/Gloo Windows not verified"
)
def test_eegclip_distributed_actual_ddp_matches_global_reference(tmp_path):
    """Validate DDP's own gradient mean, not an emulated manual all-reduce."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo distributed backend is unavailable")
    result_path = tmp_path / "real-ddp-reference.json"
    _run_with_rank_zero_in_test_process(
        _actual_ddp_worker, str(tmp_path / "real-ddp"), str(result_path)
    )
    with result_path.open(encoding="utf-8") as result:
        assert json.load(result)["max_abs_error"] < 1e-11
