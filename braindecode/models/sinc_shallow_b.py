import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SincConv2D(nn.Module):
    """
    PyTorch implementation of the SincConv2D layer.

    This layer performs convolution with filters that are parametrized using
    sinc functions, allowing for learnable band-pass filters.

    Parameters
    ----------
    N_filt : int
        Number of filters.
    Filt_dim : int
        Filter dimension (length of the filters).
    fs : int
        Sampling frequency.
    padding : str, optional
        Padding mode ('same' or 'valid'). Default is 'same'.
    """

    def __init__(self, N_filt, Filt_dim, fs, padding="same"):
        super(SincConv2D, self).__init__()
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.padding = padding.lower()

        # Initialize filter parameters
        low_freq = 4
        high_freq = 38

        # Random initialization between low_freq and high_freq
        low_hz = np.random.uniform(low_freq, high_freq, self.N_filt)
        low_hz = low_hz / (self.fs / 2)  # Normalize between 0 and 1 (Nyquist frequency)

        # Equally spaced frequency bands
        hz_points = np.linspace(low_freq, high_freq, self.N_filt + 1)
        band_hz = np.diff(hz_points)
        band_hz = band_hz / (self.fs / 2)  # Normalize

        # Convert to torch parameters
        self.filt_b1 = nn.Parameter(torch.Tensor(low_hz).view(-1, 1))
        self.filt_band = nn.Parameter(torch.Tensor(band_hz).view(-1, 1))

        # Time axis for the filters
        t_right = (
            torch.linspace(1, (self.Filt_dim - 1) / 2, steps=(self.Filt_dim - 1) // 2)
            / self.fs
        )
        self.register_buffer("t_right", t_right)

        # Window function
        n = np.linspace(0, self.Filt_dim - 1, self.Filt_dim)
        window = 0.54 - 0.46 * np.cos(2 * np.pi * n / self.Filt_dim)
        self.register_buffer("window", torch.Tensor(window.astype(np.float32)))

    def forward(self, x):
        """
        Forward pass of the SincConv2D layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after applying the sinc convolution.
        """
        # Compute filter frequencies
        min_freq = 4 / (self.fs / 2)  # Normalize
        min_band = 5 / (self.fs / 2)  # Normalize

        filt_beg_freq = min_freq + torch.abs(self.filt_b1)
        filt_end_freq = filt_beg_freq + min_band + torch.abs(self.filt_band)

        _ = (self.Filt_dim - 1) // 2
        t_right = self.t_right.to(x.device)

        filters = []
        for i in range(self.N_filt):
            # Lower and upper cutoff frequencies
            f1 = filt_beg_freq[i][0]
            f2 = torch.clamp(filt_end_freq[i][0], f1 + min_band, 1.0)

            # Sinc filters
            band = self._sinc(f2 * (self.fs / 2), t_right) - self._sinc(
                f1 * (self.fs / 2), t_right
            )

            band = band / torch.max(band)
            band = band * self.window.to(x.device)

            filters.append(band)

        filters = torch.stack(filters)  # Shape: (N_filt, Filt_dim)
        filters = filters.view(self.N_filt, 1, self.Filt_dim, 1)

        # Adjust input shape to match Conv2d requirements
        # Input x is expected to be (batch_size, channels, height, width)
        # For EEG data, we can treat channels as height and time as width
        # If x has shape (batch_size, C, T, 1), we need to permute it
        x = x.permute(0, 3, 1, 2)  # Now x is (batch_size, 1, C, T)

        # Apply padding
        if self.padding == "same":
            padding = (self.Filt_dim // 2, 0)
        elif self.padding == "valid":
            padding = (0, 0)
        else:
            raise ValueError("Unsupported padding mode: {}".format(self.padding))

        # Apply convolution
        out = F.conv2d(x, filters, stride=1, padding=padding)

        # Permute back to original shape
        out = out.permute(0, 2, 3, 1)  # Shape: (batch_size, C, T, N_filt)

        return out

    def _sinc(self, fc, t_right):
        """
        Helper function to compute sinc function.

        Parameters
        ----------
        fc : torch.Tensor
            Cutoff frequency.
        t_right : torch.Tensor
            Time axis.

        Returns
        -------
        torch.Tensor
            Sinc filter.
        """
        y_right = torch.sin(2 * math.pi * fc * t_right) / (2 * math.pi * fc * t_right)
        y_left = torch.flip(y_right, dims=[0])
        y = torch.cat([y_left, torch.tensor([1.0], device=fc.device), y_right])

        return y


class SincShallowNet(nn.Module):
    """
    PyTorch implementation of the SincShallowNet model.

    This model is designed for EEG signal classification using Sinc-based convolutions.

    Parameters
    ----------
    nb_classes : int
        Number of output classes.
    C : int
        Number of EEG channels.
    T : int
        Number of time samples.
    dropoutRate : float
        Dropout rate.
    kernLength : int
        Kernel length for the SincConv2D layer.
    F1 : int
        Number of temporal filters.
    D : int
        Depth multiplier for the depthwise convolution.
    F2 : int
        Number of pointwise convolution filters.
    norm_rate : float
        Max-norm regularization rate.
    dropoutType : str
        Type of dropout ('Dropout' or 'SpatialDropout2D').
    """

    def __init__(
        self,
        nb_classes,
        C,
        T,
        dropoutRate=0.5,
        kernLength=32,
        F1=8,
        D=2,
        dropoutType="Dropout",
    ):
        super(SincShallowNet, self).__init__()

        if dropoutType == "SpatialDropout2D":
            self.dropout = nn.Dropout2d
        elif dropoutType == "Dropout":
            self.dropout = nn.Dropout
        else:
            raise ValueError("dropoutType must be one of SpatialDropout2D or Dropout.")

        self.sinc_conv = SincConv2D(
            N_filt=F1, Filt_dim=kernLength, fs=128, padding="same"
        )
        self.batch_norm1 = nn.BatchNorm2d(F1)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            F1, F1 * D, kernel_size=(C, 1), groups=F1, bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()

        # Pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 109), stride=(1, 23))

        # Dropout
        self.drop = self.dropout(p=dropoutRate)

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(F1 * D * ((T - 109) // 23 + 1), nb_classes)
        self.fc_constraint = nn.utils.weight_norm(self.fc, dim=None)

        # Softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the SincShallowNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, C, T, 1).

        Returns
        -------
        torch.Tensor
            Output tensor with class probabilities.
        """
        # x shape: (batch_size, C, T, 1)
        x = self.sinc_conv(x)
        x = self.batch_norm1(x)
        x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)

        x = self.depthwise_conv(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.drop(x)

        x = self.flatten(x)
        x = self.fc_constraint(x)
        x = self.softmax(x)

        return x


# Instantiate the model
C = 22
T = 1001
model = SincShallowNet(
    nb_classes=4,
    C=C,
    T=T,
    dropoutRate=0.25,
    kernLength=33,
    F1=32,
    D=2,
    dropoutType="Dropout",
)

# Print the model summary
print(model)

x = torch.zeros(1, 22, 1001)

model(x)
