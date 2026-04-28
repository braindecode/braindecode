# Authors: Vandit Shah <vanditshah@outlook.com>
#
# License: BSD (3-clause)

import torch

_LAYER_REQUIRED_METHODS = {"GuidedGradCam", "LayerGradCam"}

_VIZ_EXTRAS_HINT = (
    "Attribution methods require captum. Install with "
    "`pip install braindecode[viz]`."
)


def _import_captum():
    try:
        from captum import attr as captum_attr
    except ImportError as exc:
        raise ImportError(_VIZ_EXTRAS_HINT) from exc
    return captum_attr


def attribute_image_features(model, input_tensor, target, method, layer=None):
    """Compute attribution maps for EEG input using a Captum method.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model to explain.
    input_tensor : torch.Tensor
        Input EEG tensor of shape ``(batch, n_chans, n_times)``.
    target : torch.Tensor
        Class labels for each sample in the batch.
    method : str
        Captum attribution method. One of:

        - ``"Saliency"`` — gradient magnitude w.r.t. input
        - ``"InputXGradient"`` — element-wise input × gradient
        - ``"IntegratedGradients"`` — path integral of gradients from baseline
        - ``"GuidedBackprop"`` — backpropagation through ReLU positive activations
        - ``"Deconvolution"`` — DeconvNet-style backward pass
        - ``"DeepLift"`` — contribution scores via activation differences
        - ``"LRP"`` — Layer-wise Relevance Propagation
        - ``"GuidedGradCam"`` — GuidedBackprop × GradCAM (requires ``layer``)
        - ``"LayerGradCam"`` — class activation maps at a specific layer
          (requires ``layer``)
    layer : torch.nn.Module, optional
        Target layer for ``GuidedGradCam`` / ``LayerGradCam``.

    Returns
    -------
    numpy.ndarray
        Attribution map of the same spatial shape as ``input_tensor``.
    """
    if method in _LAYER_REQUIRED_METHODS and layer is None:
        raise ValueError(
            f"method='{method}' requires a `layer` argument; pass the target "
            "module (e.g. the last convolutional layer)."
        )

    captum_attr = _import_captum()
    model.zero_grad()
    algorithm = get_attribution_method(model, method, layer=layer)

    if method == "LayerGradCam":
        # Captum applies ReLU to LayerGradCam by default; pass False to keep
        # signed activations so downstream metrics see both signs.
        attributions = algorithm.attribute(
            input_tensor, target=target, relu_attributions=False
        )
        attributions = captum_attr.LayerAttribution.interpolate(
            attributions, input_tensor.shape[-2:]
        )
    else:
        attributions = algorithm.attribute(input_tensor, target=target)
        if method == "GuidedGradCam" and attributions.numel() == 0:
            raise ValueError(
                "GuidedGradCam returned empty attributions, typically because "
                "the model changes input dimensionality internally (e.g. via "
                "Ensure4d). Use a model that keeps consistent input/layer dims."
            )

    return attributions.cpu().detach().numpy().squeeze()


def get_attributions(model, X, y, method, layer=None, device="cpu"):
    """Compute attribution maps for correctly classified EEG samples.

    Runs inference on the full dataset, filters to correctly classified
    samples, then computes attribution maps.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model to explain.
    X : numpy.ndarray
        EEG data of shape ``(n_samples, n_chans, n_times)``.
    y : numpy.ndarray
        True class labels of shape ``(n_samples,)``.
    method : str
        Captum attribution method. See :func:`attribute_image_features`.
    layer : torch.nn.Module, optional
        Target layer for layer-wise methods.
    device : str, default="cpu"
        Device to run inference on.

    Returns
    -------
    attributions : numpy.ndarray
        Attribution maps of shape ``(n_correct, n_chans, n_times)``.
    labels : numpy.ndarray
        True labels for the correctly classified samples.
    """
    model.eval()
    model.to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    with torch.no_grad():
        preds = model(X_tensor).argmax(dim=1)

    correct = preds == y_tensor
    X_correct = X_tensor[correct].requires_grad_(True)
    y_correct = y_tensor[correct]

    attributions = attribute_image_features(
        model, X_correct, y_correct, method, layer=layer
    )
    return attributions, y_correct.cpu().numpy()


def get_attribution_method(model, method, layer=None):
    """Factory for Captum attribution method instances.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model to explain.
    method : str
        Captum attribution method name. See :func:`attribute_image_features`.
    layer : torch.nn.Module, optional
        Target layer; required for ``GuidedGradCam`` and ``LayerGradCam``,
        unused otherwise.

    Returns
    -------
    captum.attr._utils.attribution.Attribution
        Instantiated Captum attribution object.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported method names, or if
        ``layer`` is required but not provided.
    """
    captum_attr = _import_captum()

    if method in _LAYER_REQUIRED_METHODS and layer is None:
        raise ValueError(
            f"method='{method}' requires a `layer` argument; pass the target "
            "module (e.g. the last convolutional layer)."
        )

    methods = {
        "Saliency": captum_attr.Saliency,
        "InputXGradient": captum_attr.InputXGradient,
        "IntegratedGradients": captum_attr.IntegratedGradients,
        "GuidedBackprop": captum_attr.GuidedBackprop,
        "Deconvolution": captum_attr.Deconvolution,
        "DeepLift": captum_attr.DeepLift,
        "LRP": captum_attr.LRP,
        "GuidedGradCam": lambda m: captum_attr.GuidedGradCam(m, layer),
        "LayerGradCam": lambda m: captum_attr.LayerGradCam(m, layer),
    }

    if method not in methods:
        raise ValueError(
            f"Unsupported attribution method: '{method}'. "
            f"Choose one of: {sorted(methods.keys())}"
        )

    return methods[method](model)
