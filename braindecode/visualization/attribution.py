# Authors: Vandit Shah <vanditshah@outlook.com>
#
# License: BSD (3-clause)

import torch
from captum.attr import (
    LRP,
    Deconvolution,
    DeepLift,
    GuidedBackprop,
    GuidedGradCam,
    InputXGradient,
    IntegratedGradients,
    LayerGradCam,
    Saliency,
)
from pytorch_grad_cam import FullGrad, GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_CAM_METHODS = {"GradCAM", "GradCAMPlusPlus", "ScoreCAM", "LayerCAM", "FullGrad"}


def attribute_image_features(model, input_tensor, target, method, layer=None):
    """Compute attribution maps for EEG input using a specified method.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model to explain.
    input_tensor : torch.Tensor
        Input EEG tensor of shape ``(batch, n_chans, n_times)``.
    target : torch.Tensor
        Class labels for each sample in the batch.
    method : str
        Attribution method name. One of:

        **Captum methods:**

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

        **CAM methods** (require ``layer``):

        - ``"GradCAM"`` — gradient-weighted class activation maps
        - ``"GradCAMPlusPlus"`` — improved GradCAM with higher-order gradients
        - ``"ScoreCAM"`` — perturbation-based CAM without gradients
        - ``"LayerCAM"`` — spatial class activation maps per layer
        - ``"FullGrad"`` — full-gradient saliency maps
    layer : torch.nn.Module, optional
        Target layer for layer-wise methods. Required for CAM methods and
        ``"GuidedGradCam"``, ``"LayerGradCam"``.

    Returns
    -------
    numpy.ndarray
        Attribution map of the same spatial shape as ``input_tensor``.
    """
    model.zero_grad()

    if method in _CAM_METHODS:
        # pytorch-grad-cam requires at least 4D input (batch, C, H, W).
        # EEG inputs are 3D (batch, n_chans, n_times); unsqueeze to
        # (batch, n_chans, n_times, 1) which matches what Ensure4d produces
        # internally, so the model forward pass is unaffected.
        needs_unsqueeze = input_tensor.ndim == 3
        if needs_unsqueeze:
            input_tensor = input_tensor.unsqueeze(-1)
        targets = [ClassifierOutputTarget(t.item()) for t in target]
        algorithm = get_cam_method(model, method, target_layers=[layer])
        attributions = algorithm(input_tensor=input_tensor, targets=targets)
        # Output is (batch, n_times, 1) for 4D input; drop the trailing dim.
        if needs_unsqueeze and attributions.ndim == 3:
            attributions = attributions[..., 0]
        return attributions
    else:
        from captum.attr import LayerAttribution

        algorithm = get_attribution_method(model, method, layer=layer)
        if method == "LayerGradCam":
            attributions = algorithm.attribute(
                input_tensor, target=target, relu_attributions=False
            )
            attributions = LayerAttribution.interpolate(
                attributions, input_tensor.shape[-2:]
            )
        else:
            attributions = algorithm.attribute(input_tensor, target=target)
            if method == "GuidedGradCam" and attributions.numel() == 0:
                raise ValueError(
                    "GuidedGradCam returned empty attributions. This typically "
                    "happens when the model internally changes the input "
                    "dimensionality (e.g. via Ensure4d), causing a spatial "
                    "dimension mismatch during interpolation. Use a model that "
                    "keeps consistent input/layer dimensions."
                )

        return attributions.cpu().detach().numpy().squeeze()


def get_attributions(model, X, y, method, layer=None, device="cpu"):
    """Compute attribution maps for correctly classified EEG samples.

    Runs inference on the full dataset, filters to correctly classified
    samples, then computes attribution maps for each sample.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model to explain.
    X : numpy.ndarray
        EEG data of shape ``(n_samples, n_chans, n_times)``.
    y : numpy.ndarray
        True class labels of shape ``(n_samples,)``.
    method : str
        Attribution method name. See :func:`attribute_image_features` for
        supported values.
    layer : torch.nn.Module, optional
        Target layer for layer-wise methods.
    device : str, default="cpu"
        Device to run inference on (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    attributions : numpy.ndarray
        Attribution maps of shape ``(n_correct, n_chans, n_times)``.
    labels : numpy.ndarray
        True labels for the correctly classified samples.
    """
    model.eval()
    model.to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        preds = model(X_tensor).argmax(dim=1)

    correct = preds == y_tensor
    X_correct = X_tensor[correct]
    y_correct = y_tensor[correct]

    X_correct.requires_grad_(True)
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
        Attribution method name. One of:

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
        Target layer for layer-wise methods. Falls back to the last child
        module of ``model`` if not provided.

    Returns
    -------
    captum.attr._utils.attribution.Attribution
        Instantiated Captum attribution object.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported method names.
    """
    if layer is None:
        layer = list(model.children())[-1]

    methods = {
        "Saliency": lambda: Saliency(model),
        "InputXGradient": lambda: InputXGradient(model),
        "IntegratedGradients": lambda: IntegratedGradients(model),
        "GuidedBackprop": lambda: GuidedBackprop(model),
        "Deconvolution": lambda: Deconvolution(model),
        "DeepLift": lambda: DeepLift(model),
        "LRP": lambda: LRP(model),
        "GuidedGradCam": lambda: GuidedGradCam(model, layer),
        "LayerGradCam": lambda: LayerGradCam(model, layer),
    }

    if method not in methods:
        raise ValueError(
            f"Unsupported attribution method: '{method}'. "
            f"Choose one of: {sorted(methods.keys())}"
        )

    return methods[method]()


def get_cam_method(model, method, target_layers):
    """Factory for pytorch-grad-cam CAM method instances.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model to explain.
    method : str
        CAM method name. One of:

        - ``"GradCAM"`` — gradient-weighted class activation maps
        - ``"GradCAMPlusPlus"`` — improved GradCAM with higher-order gradients
        - ``"ScoreCAM"`` — perturbation-based CAM without gradients
        - ``"LayerCAM"`` — spatial class activation maps per layer
        - ``"FullGrad"`` — full-gradient saliency maps
    target_layers : list of torch.nn.Module
        List of layers to compute CAM activations from. Typically the last
        convolutional layer, e.g. ``[model.conv5]``.

    Returns
    -------
    pytorch_grad_cam.base_cam.BaseCAM
        Instantiated CAM object ready to call with input tensors.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported CAM method names.
    """
    methods = {
        "GradCAM": lambda: GradCAM(model=model, target_layers=target_layers),
        "GradCAMPlusPlus": lambda: GradCAMPlusPlus(
            model=model, target_layers=target_layers
        ),
        "ScoreCAM": lambda: ScoreCAM(model=model, target_layers=target_layers),
        "LayerCAM": lambda: LayerCAM(model=model, target_layers=target_layers),
        "FullGrad": lambda: FullGrad(model=model, target_layers=target_layers),
    }

    if method not in methods:
        raise ValueError(
            f"Unsupported CAM method: '{method}'. "
            f"Choose one of: {sorted(methods.keys())}"
        )

    return methods[method]()
