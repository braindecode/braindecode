# Authors: Akshay Sujatha Ravindran <asujatharavindran@uh.edu>  (adapted
#          from the author's research code for Sujatha Ravindran &
#          Contreras-Vidal, "An empirical comparison of deep learning
#          explainability approaches for EEG using simulated ground
#          truth," Scientific Reports 13, 2023.
#          DOI: 10.1038/s41598-023-43871-8)
#
# License: BSD (3-clause)

"""Sanity-check protocols for attribution methods.

Implements the model- and label-randomization checks from Adebayo et al.
(2018) "Sanity Checks for Saliency Maps" (NeurIPS) as applied to EEG
decoders by Sujatha Ravindran & Contreras-Vidal (Sci Rep 2023):

- :func:`random_target` swaps each true class label for a different one
  (label-randomization check). An attribution method whose maps are
  unchanged when the asked-about class changes is suspicious — it isn't
  class-discriminative.
- :func:`cascading_layer_reset` yields model copies with progressively
  randomized parameters from output to input
  (cascading-weight-randomization check). An attribution method whose
  maps survive every cascade level is suspicious — its output likely
  depends on architecture rather than on learned weights.
"""

import copy

import numpy as np
import torch


def random_target(target, n_classes, generator=None):
    """Return labels uniformly sampled from ``{0, ..., n_classes-1} \\ target``.

    For each entry of ``target`` pick a different class at random.
    Used in the label-randomization sanity check: query the trained
    model's attribution method with the wrong target and check whether
    the resulting map differs from the correct-target map. Accepts a
    Python int, NumPy array, or torch tensor and returns the same kind
    of object on the same device.

    Parameters
    ----------
    target : int, numpy.ndarray, or torch.Tensor
        True class index/indices.
    n_classes : int
        Total number of classes; must be at least 2.
    generator : numpy.random.Generator, optional
        Source of randomness. Defaults to ``numpy.random.default_rng()``.

    Returns
    -------
    Same type as ``target`` (or ``int`` when ``target`` is a scalar).
    """
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2 to pick a different class.")
    rng = generator if generator is not None else np.random.default_rng()

    is_tensor = isinstance(target, torch.Tensor)
    true_classes = target.detach().cpu().numpy() if is_tensor else np.asarray(target)

    # Sample in [0, n_classes-2], then shift values >= true past true. The
    # result is a uniform draw over {0, ..., n_classes-1} \ {true}.
    drawn = rng.integers(low=0, high=n_classes - 1, size=true_classes.shape)
    drawn = np.where(drawn >= true_classes, drawn + 1, drawn)

    if is_tensor:
        return torch.as_tensor(drawn, dtype=target.dtype, device=target.device)
    if drawn.ndim == 0:
        return int(drawn)
    return drawn.astype(true_classes.dtype, copy=False)


def cascading_layer_reset(model, deepcopy_first=True):
    """Yield model copies with progressively-randomized parameters.

    Walks the modules in *reverse* depth-first order (output → input)
    and yields ``(layer_name, model_copy)`` after each layer's
    ``reset_parameters`` call. Implements the cascading-randomization
    sanity check from Adebayo et al. (NeurIPS 2018), as applied to EEG
    decoders in Sujatha Ravindran & Contreras-Vidal (Sci Rep 2023).
    An attribution method whose maps survive every cascade level is
    suspicious — its output likely depends on architecture rather than
    on learned weights.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model. The original model is not modified when
        ``deepcopy_first=True`` (default).
    deepcopy_first : bool
        If True, deep-copy the model before mutating, so the caller's
        ``model`` stays trained. Set False to skip the copy when memory
        is tight and the caller doesn't need the trained weights again.

    Yields
    ------
    layer_name : str
        Dotted-path name of the layer just reset (e.g. ``"final_layer.conv_classifier"``).
    randomized_model : torch.nn.Module
        A model whose layers from this one back to the output are
        randomly re-initialized.
    """
    target = copy.deepcopy(model) if deepcopy_first else model

    # Modules whose `reset_parameters` is callable, skipping the root
    # (which we never reset, even if the model class defines it).
    resettable = [
        (name, module)
        for name, module in target.named_modules()
        if module is not target and callable(getattr(module, "reset_parameters", None))
    ]

    for name, module in reversed(resettable):
        module.reset_parameters()
        yield name, target
