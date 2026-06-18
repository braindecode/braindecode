# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
#
# ============================================================================
# TEMPORARY -- DELETE WHEN UPSTREAM RE-HOSTS WEIGHTS
# ----------------------------------------------------------------------------
# Converts the *currently released* MVPFormer ("GeNIE") checkpoints (raw
# PyTorch state dicts, IBM Box) onto braindecode's `MVPFormer` parameter names.
# Once the authors re-publish braindecode-native weights, remove this whole
# file (it is the only place that knows the upstream key scheme).
#
# Observed in the released Sept/Oct-2024 checkpoints (genie-{s,m}-{base,swec}.pt):
#   * backbone prefix is ``genie.`` (the public code later renamed it
#     ``mvpformer.``; both are accepted here);
#   * each attention block carries vestigial ``time_bias`` / ``masked_bias``
#     buffers and a dead ``ln_2`` -- all dropped;
#   * the ``base`` head ``head.head`` is the (d_model, d_model) *generative*
#     projection, not a classifier -- dropped (attach a fresh head / reset_head);
#   * ``swec`` files contain only LoRA deltas + the real classification head
#     (no backbone), so a seizure classifier = base backbone + merged LoRA +
#     swec head.
# ============================================================================

from __future__ import annotations


# ponytail: keep all upstream-key knowledge in this one disposable file.

_BACKBONE_PREFIXES = ("genie.", "mvpformer.")


def _strip_backbone_prefix(key: str) -> str:
    for prefix in _BACKBONE_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _is_dropped(key: str) -> bool:
    """Upstream keys with no braindecode counterpart."""
    return (
        key == "seizure_embeddings"  # generative-only
        or ".ln_2." in key  # instantiated upstream but never used
        or key.endswith(".time_bias")  # vestigial attention buffer
        or key.endswith(".masked_bias")  # vestigial attention buffer
        or ".lora_" in key  # LoRA deltas (merged separately for swec)
        or key.startswith("head.")  # generative projection; classifier via swec
    )


def _map_backbone_key(key: str) -> str | None:
    """Map one upstream *backbone* key to its braindecode name (or ``None``)."""
    if _is_dropped(key):
        return None
    if key.startswith("encoder."):
        return "patch_embed." + key[len("encoder.") :]
    rest = _strip_backbone_prefix(key)  # genie./mvpformer. -> ""
    if rest.startswith("h."):
        return "blocks." + rest[len("h.") :]
    return rest  # ln_f.weight, positional_embedding.weight, channel_embedding.weight


def convert_mvpformer_state_dict(state_dict: dict, kind: str = "base") -> dict:
    """Convert a raw upstream MVPFormer/GeNIE checkpoint to ``MVPFormer`` keys.

    .. warning::
        Temporary shim for the currently released checkpoints; see the module
        banner. It will be removed once braindecode-native weights are
        published.

    Parameters
    ----------
    state_dict : dict
        Raw ``torch.load`` of an upstream ``*-base.pt`` checkpoint.
    kind : {"base"}
        Only ``"base"`` (the generatively pre-trained, LoRA-free backbone) is
        supported here. A full ``swec`` seizure classifier is built with
        :func:`merge_swec_checkpoint` (base backbone + LoRA + swec head).

    Returns
    -------
    dict
        State dict with ``MVPFormer`` parameter names. Load with
        ``model.load_state_dict(converted, strict=False)``; the fresh
        classification head appears as the only missing key (expected for a
        backbone) -- call :meth:`MVPFormer.reset_head` or train it.
    """
    if kind != "base":
        raise ValueError(
            f"convert_mvpformer_state_dict only handles kind='base', got {kind!r}. "
            "Use merge_swec_checkpoint for swec classifiers."
        )
    converted = {}
    for key, value in state_dict.items():
        new_key = _map_backbone_key(key)
        if new_key is not None:
            converted[new_key] = value
    return converted


def merge_swec_checkpoint(
    base_state_dict: dict,
    swec_state_dict: dict,
    lora_alpha: int = 16,
    lora_rank: int = 8,
) -> dict:
    """Build a full MVPFormer seizure classifier from a base backbone + swec.

    .. warning::
        Temporary shim for the currently released checkpoints; see the module
        banner.

    The released ``swec`` files contain only LoRA deltas (on ``q_attn`` and
    ``c_attn``, as plain ``loralib.Linear`` adapters -- verified against
    ``loralib``) plus the classification head. This merges
    ``W <- W + (lora_alpha / lora_rank) * (lora_B @ lora_A)`` into the backbone
    and installs the swec head as ``final_layer``.

    Parameters
    ----------
    base_state_dict : dict
        Raw upstream ``*-base.pt`` checkpoint (the backbone).
    swec_state_dict : dict
        Raw upstream ``*-swec.pt`` checkpoint (LoRA deltas + head).
    lora_alpha, lora_rank : int
        LoRA scaling parameters (released config: 16 and 8 -> scaling 2).

    Returns
    -------
    dict
        Full classifier state dict in ``MVPFormer`` naming;
        ``model.load_state_dict(merged)`` should match exactly (use a model with
        ``n_outputs`` equal to the swec head, e.g. 2 for seizure detection).
    """
    converted = convert_mvpformer_state_dict(base_state_dict, kind="base")
    scaling = lora_alpha / lora_rank

    for key in swec_state_dict:
        if not key.endswith(".lora_A"):
            continue
        stem = key[: -len(".lora_A")]  # e.g. genie.h.0.attn.q_attn
        lora_A = swec_state_dict[key]
        lora_B = swec_state_dict[stem + ".lora_B"]
        target = _map_backbone_key(stem + ".weight")  # blocks.{i}.attn.*_attn.weight
        if target is None or target not in converted:
            raise KeyError(f"LoRA target for {key!r} ({target}) not in backbone.")
        converted[target] = converted[target] + scaling * (lora_B @ lora_A)

    head = swec_state_dict.get("head.head.weight")
    if head is not None:
        converted["final_layer.weight"] = head
    return converted
