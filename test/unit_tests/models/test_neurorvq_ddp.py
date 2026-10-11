# License: BSD-3
"""NeuroRVQ cold-start consistency across independent distributed ranks."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from braindecode.models.neurorvq_tokenizer import _EMAVectorQuantizer


def _distributed_cold_start(rank, world_size, init_file, out_dir):
    """Use distinct rank-local inputs, then compare actual quantizer states."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1234 + rank)
        quantizer = _EMAVectorQuantizer(n_codes=8, code_dim=4).train()
        # DDP's initial module broadcast happens *before* the first forward;
        # these deliberately different first batches stress the later
        # data-dependent in-forward codebook initialization.
        z = torch.randn(2, 4, 2, 4) + 2.0 * rank
        vectors = F.normalize(z.permute(0, 2, 3, 1), dim=-1).reshape(-1, 4)

        quantizer.embedding.initialize(vectors)
        initialized = {
            "weight": quantizer.embedding.weight.detach().clone(),
            "cluster_size": quantizer.embedding.cluster_size.detach().clone(),
            "embed_avg": quantizer.embedding.embed_avg.detach().clone(),
            "initted": quantizer.embedding.initted.detach().clone(),
        }
        # Training must subsequently all-reduce counts/sums, starting from
        # identical centroids and thus a coherent token-to-code mapping.
        quantizer(z)
        updated = {
            "weight": quantizer.embedding.weight.detach().clone(),
            "cluster_size": quantizer.embedding.cluster_size.detach().clone(),
            "embed_avg": quantizer.embedding.embed_avg.detach().clone(),
            "initted": quantizer.embedding.initted.detach().clone(),
            "usage_cluster_size": quantizer.cluster_size.detach().clone(),
        }
        torch.save(
            {"initialized": initialized, "updated": updated},
            Path(out_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo torch.distributed is unavailable",
)
def test_neurorvq_cold_codebook_ddp_rank_consistency(tmp_path):
    """Rank-local seeds/batches must not result in separate codebook states."""
    mp.spawn(
        _distributed_cold_start,
        args=(2, str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    states = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for rank in (0, 1)
    ]
    for phase in ("initialized", "updated"):
        for name, value in states[0][phase].items():
            torch.testing.assert_close(value, states[1][phase][name], rtol=0, atol=0)
    assert states[0]["initialized"]["initted"].item() == 1
    assert states[0]["updated"]["initted"].item() == 1
    assert states[0]["updated"]["usage_cluster_size"].sum().item() > 0


class _DDPWrappedTokenizer(torch.nn.Module):
    """Exercise lazy quantizer init inside an actual DDP forward/backward."""

    def __init__(self):
        super().__init__()
        # DDP requires at least one trainable parameter, while NeuroRVQ's
        # EMA codebook parameters deliberately do not require gradients.
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.quantizer = _EMAVectorQuantizer(n_codes=8, code_dim=4)

    def forward(self, x):
        quantized, _ = self.quantizer(x * self.scale)
        return quantized


def _distributed_ddp_wrapped_forward(rank, world_size, init_file, out_dir):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(177 + rank)
        wrapped = torch.nn.parallel.DistributedDataParallel(
            _DDPWrappedTokenizer().train()
        )
        # DDP synchronizes initial parameters/buffers on construction,
        # not the rank-dependent K-means centroids created inside forward.
        torch.manual_seed(991 + rank)
        x = torch.randn(2, 4, 2, 4) + 3.0 * rank
        wrapped(x).mean().backward()

        emb = wrapped.module.quantizer.embedding
        torch.save(
            {
                "weight": emb.weight.detach().clone(),
                "cluster_size": emb.cluster_size.detach().clone(),
                "initted": emb.initted.detach().clone(),
                "usage_cluster_size": (
                    wrapped.module.quantizer.cluster_size.detach().clone()
                ),
                "scale_grad": wrapped.module.scale.grad.detach().clone(),
            },
            Path(out_dir) / f"ddp-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo torch.distributed is unavailable",
)
def test_neurorvq_cold_codebook_with_actual_ddp_wrapper(tmp_path):
    """DDP's construction-time broadcast must not mask lazy-init divergence."""
    mp.spawn(
        _distributed_ddp_wrapped_forward,
        args=(2, str(tmp_path / "ddp-init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    states = [
        torch.load(tmp_path / f"ddp-rank-{rank}.pt", weights_only=True)
        for rank in (0, 1)
    ]
    for key, first_rank in states[0].items():
        torch.testing.assert_close(first_rank, states[1][key], rtol=0, atol=0)
    assert states[0]["initted"].item() == 1
    assert states[0]["usage_cluster_size"].sum().item() > 0


def test_neurorvq_cold_codebook_single_process_does_not_require_dist():
    """The same fix must not change ordinary CPU first-batch initialization."""
    quantizer = _EMAVectorQuantizer(n_codes=8, code_dim=4).train()
    torch.manual_seed(1)
    z = torch.randn(2, 4, 2, 4)
    y, ids = quantizer(z)
    assert y.shape == z.shape
    assert ids.numel() == 16
    assert quantizer.embedding.initted.item() == 1
    assert torch.isfinite(quantizer.embedding.weight).all()
    previous = quantizer.embedding.weight.detach().clone()
    quantizer.eval()
    quantizer.encode(z)
    torch.testing.assert_close(quantizer.embedding.weight, previous)
