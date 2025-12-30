"""Independent Component Analysis (ICA) for artifact removal.

Wrapper for MNE's ICA implementation to remove ocular and other artifacts.
"""

# Authors: [Your Name]
# License: BSD (3-clause)

from typing import Optional, List, Union
import warnings

import mne
from mne.preprocessing import ICA as MNE_ICA

from .preprocess import Preprocessor


class ICA(Preprocessor):
    """Independent Component Analysis for artifact removal.

    Fits ICA on the raw or epochs data and removes selected components.
    Can automatically detect ocular artifacts using EOG channels or accept
    manual component exclusion.

    Parameters
    ----------
    n_components : int | float | None
        Number of principal components used for ICA decomposition.
        If int, specifies the number of components.
        If float (0 < n_components < 1), selects the number of components
        explaining the given fraction of variance.
        If None, all components are used (default).
    method : str
        ICA method to use. Options: 'fastica', 'infomax', 'picard'.
        Default is 'fastica'.
    random_state : int | None
        Seed for the random number generator for reproducibility.
    eog_channels : list of str | None
        List of channel names to use for EOG correlation. If provided,
        components with correlation above threshold (default 0.3) will be
        automatically excluded.
    exclude : list of int | None
        Manually specify component indices to exclude. Overrides automatic
        detection.
    fit_params : dict | None
        Additional parameters passed to ICA.fit().
    apply_params : dict | None
        Additional parameters passed to ICA.apply().
    picks : str | list | slice | None
        Channels to include in ICA. Defaults to 'eeg'.
    **kwargs : dict
        Additional keyword arguments forwarded to the MNE ICA constructor.
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        method: str = "fastica",
        random_state: Optional[int] = None,
        eog_channels: Optional[List[str]] = None,
        exclude: Optional[List[int]] = None,
        fit_params: Optional[dict] = None,
        apply_params: Optional[dict] = None,
        picks: Optional[Union[str, list, slice]] = "eeg",
        **kwargs,
    ):
        # Store parameters for serialization
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.eog_channels = eog_channels
        self.exclude = exclude
        self.fit_params = fit_params or {}
        self.apply_params = apply_params or {}
        self.picks = picks
        self.ica_kwargs = kwargs

        # We will not use the generic Preprocessor's fn mechanism.
        # Instead we override apply.
        super().__init__(fn=self._apply_ica, apply_on_array=False)

    def _apply_ica(self, raw_or_epochs):
        """Internal function that performs ICA fit and apply."""
        # Create ICA instance
        ica = MNE_ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            **self.ica_kwargs,
        )

        # Fit ICA
        ica.fit(raw_or_epochs, picks=self.picks, **self.fit_params)

        # Determine components to exclude
        exclude = self.exclude if self.exclude is not None else []

        if self.eog_channels is not None:
            # Find EOG artifacts
            eog_indices, eog_scores = ica.find_bads_eog(
                raw_or_epochs,
                ch_name=self.eog_channels,
                threshold=0.3,  # default threshold
            )
            exclude.extend(eog_indices)
            if eog_indices:
                msg = f"Excluding EOG-related components: {eog_indices}"
                warnings.warn(msg)

        # Remove duplicates
        exclude = list(set(exclude))

        if exclude:
            # Apply ICA, removing the excluded components
            ica.apply(raw_or_epochs, exclude=exclude, **self.apply_params)
        else:
            # No components to remove; still apply with empty exclude
            ica.apply(raw_or_epochs, exclude=[], **self.apply_params)

        # Return None because ICA.apply modifies raw_or_epochs inplace
        return None

    @property
    def _all_attrs(self):
        """Return all attributes for serialization."""
        base_attrs = super()._all_attrs
        extra = [
            "n_components",
            "method",
            "random_state",
            "eog_channels",
            "exclude",
            "fit_params",
            "apply_params",
            "picks",
            "ica_kwargs",
        ]
        return base_attrs + extra

    def __repr__(self):
        """Representation of the ICA preprocessor."""
        params = []
        for attr in self._all_attrs:
            if attr in ("fn", "apply_on_array", "kwargs"):
                continue
            val = getattr(self, attr, None)
            params.append(f"{attr}={val}")
        return f"ICA({', '.join(params)})"