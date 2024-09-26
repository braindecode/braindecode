from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from torch import FloatTensor
from torch.nn.parameter import Parameter
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class GraphConvolution(nn.Module):
    """GNN from [kipfwelling2017]_.

    Applies a graph convolution operation as described in Kipf & Welling (2017).

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Number of output features per node.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: True.
    activation: nn.Module, optional
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    gain: float, optional


    References
    ----------
    .. [kipfwelling2017] Kipf, T. N., & Welling, M. (2017). Semi-supervised
        classification with graph convolutional networks. In International
        Conference on Learning Representations.
        https://openreview.net/forum?id=SJU4ayYgl
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: nn.Module = nn.ReLU,
        gain: float = 1.414,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(FloatTensor(in_features, out_features))

        self.activation = activation()
        self.gain = gain

        if bias:
            self.bias = Parameter(torch.zeros(1, 1, out_features))
        else:
            self.register_parameter("bias", None)

        # Initialization
        nn.init.xavier_uniform_(self.weight, gain=self.gain)

    def forward(self, x, adj):
        """Forward pass of the GCN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, num_nodes, in_features).
        adj : torch.Tensor
            Adjacency matrix tensor of shape (batch_size, num_nodes, num_nodes).

        Returns
        -------
        torch.Tensor
            Output feature tensor of shape (batch_size, num_nodes, out_features).
        """
        support = torch.matmul(x, self.weight)

        if self.bias is not None:
            support = support - self.bias

        output = self.activation(torch.matmul(adj, support))

        return output


class PowerLayer(nn.Module):
    """Computes the logarithm of the average power over a window.

    Parameters
    ----------
    window_length : int
        Length of the averaging window.
    step_size : int
        Step size for the averaging window.
    """

    def __init__(self, window_length, step_size):
        super().__init__()
        self.pooling = nn.AvgPool2d(
            kernel_size=(1, window_length), stride=(1, step_size)
        )

    def forward(self, x):
        """Forward pass of the PowerLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, 1, time_samples).

        Returns
        -------
        torch.Tensor
            Output tensor after applying log power computation.
        """
        power = x.pow(2)

        averaged_power = self.pooling(power)
        log_power = torch.log(averaged_power)
        return log_power


class Aggregator:
    """Aggregates features from different brain areas.

    Parameters
    ----------
    idx_area : list of int
        List containing the number of channels in each brain area.
    """

    def __init__(self, idx_area: list[int]):
        self.idx = self._get_indices(idx_area)
        self.num_areas = len(idx_area)

    def _get_indices(self, idx_area):
        indices = [0]
        for count in idx_area:
            indices.append(indices[-1] + count)
        return indices

    def forward(self, x):
        """Aggregates features by averaging over channels in each brain area.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, features).

        Returns
        -------
        torch.Tensor
            Aggregated tensor of shape (batch_size, num_areas, features).
        """
        aggregated_features = []
        for i in range(self.num_areas):
            start_idx = self.idx[i]
            end_idx = self.idx[i + 1]
            area_features = x[:, start_idx:end_idx, :]
            mean_features = torch.mean(area_features, dim=1)
            aggregated_features.append(mean_features)
        return torch.stack(aggregated_features, dim=1)


class LGGNet(EEGModuleMixin, nn.Module):
    """LGGNet model for EEG signal classification from [li2023]_.

    The LGGNet model combines temporal convolution, power layer,
    local graph filtering, and global graph convolutional networks
    to classify EEG signals.

    Parameters
    ----------
    num_T : int
        Number of temporal convolution filters.
    out_graph : int
        Number of output features from the graph convolution layer.
    dropout_rate : float
        Dropout rate for the fully connected layer.
    pool_size : int
        Pooling window size for the power layer.
    pool_step_rate : float
        Step rate for the pooling window (as a fraction of the pool size).
    idx_graph : list of int
        List containing the number of channels in each brain area.

    References
    ----------
    .. [li2023] Li, K., Gu, Z., Liao, Y., Yu, T., Li, Y., & Jin, J. (2023).
        LGGNet: A Local and Global Graph Network for EEG Classification.
        IEEE Transactions on Neural Systems and Rehabilitation Engineering.
    .. [li2023code] Li, K., Gu, Z., Liao, Y., Yu, T., Li, Y., & Jin, J. (2023).
        LGGNet: A Local and Global Graph Network for EEG Classification.
        https://github.com/yi-ding-cs/LGG
    """

    def __init__(
        self,
        # Model parameters
        idx_graph: list[int],
        num_T: int = 32,
        out_graph: int = 64,
        dropout_rate: float = 0.5,
        pool_size: int = 4,
        pool_step_rate: float = 0.5,
        activation_1: nn.Module = nn.LeakyReLU,
        activation_2: nn.Module = nn.ReLU,
        # braindecode parameters
        n_outputs=None,
        n_freqs=None,
        n_chans=None,
        n_times=None,
        sfreq=None,
        chs_info=None,
        input_window_seconds=None,
    ):
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds

        if idx_graph is None:
            raise ValueError(
                "idx_graph must be provided as a "
                "list of channel counts per brain area."
            )

        self.idx_graph = idx_graph
        self.num_areas = len(idx_graph)

        self.pool_size = pool_size
        self.window_sizes = [0.5, 0.25, 0.125]  # in seconds
        self.activation_1 = activation_1
        self.activation_2 = activation_2()

        # Temporal convolution layers
        self.temporal_layers = nn.ModuleList(
            [
                self._temporal_layer(
                    n_freqs,
                    num_T,
                    int(window_size * self.sfreq),
                    pool_size,
                    pool_step_rate,
                )
                for window_size in self.window_sizes
            ]
        )

        self.bn_temporal = nn.BatchNorm2d(num_T)
        self.bn_temporal_post = nn.BatchNorm2d(num_T)

        # Matching dimensionality
        self.ensuredim_input = Rearrange("batch channel time -> batch 1 channel time")
        self.ensuredim_graph = Rearrange("b t c time -> b c (t time)")
        self.ensuredim_flatten = Rearrange("b ... -> b (...)")

        self.one_by_one_conv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1)),
            activation_1(),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        # Local filter parameters
        self.local_filter_weight = Parameter(FloatTensor(self.n_chans, num_T))
        self.local_filter_bias = Parameter(torch.zeros(1, self.n_chans, 1))
        # Aggregator
        self.aggregator = Aggregator(self.idx_graph)
        # Global adjacency matrix (learnable)
        self.global_adj = Parameter(FloatTensor(self.num_areas, self.num_areas))

        # Batch normalization layers
        self.bn_aggregated = nn.BatchNorm1d(self.num_areas)
        self.bn_gcn = nn.BatchNorm1d(self.num_areas)

        # Graph Convolutional Network
        self.gcn = GraphConvolution(num_T, out_graph, activation=activation_2)

        # Fully connected layer
        self.final_layer = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_areas * out_graph, self.n_outputs),
        )

        self.initialize_parameters()

    def _temporal_layer(
        self, in_channels, out_channels, kernel_size, pool_size, pool_step_rate
    ):
        """Creates a temporal learning layer."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size)),
            PowerLayer(
                window_length=pool_size, step_size=int(pool_step_rate * pool_size)
            ),
        )

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.local_filter_weight)
        nn.init.xavier_uniform_(self.global_adj)

    def forward(self, x):
        """Forward pass of the LGGNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_freqs, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        # Apply temporal convolution layers
        x = self.ensuredim_input(x)  # Shape: (batch_size, 1, n_chans, n_times)
        temporal_outputs = [layer(x) for layer in self.temporal_layers]
        out = torch.cat(temporal_outputs, dim=-1)
        out = self.bn_temporal(out)
        out = self.one_by_one_conv(out)
        out = self.bn_temporal_post(out)

        # Reshape and apply local filtering
        out = self.ensuredim_graph(out)
        out = self._local_filter(out, self.local_filter_weight)

        # Aggregate features
        out = self.aggregator.forward(out)

        # Compute adjacency matrix
        adj = self._compute_adj(out)

        # Apply batch normalization and GCN
        out = self.bn_aggregated(out)
        out = self.gcn(out, adj)
        out = self.bn_gcn(out)

        # Classification
        out = self.ensuredim_flatten(out)
        out = self.final_layer(out)
        return out

    def _local_filter(self, x, weights):
        """Applies local filtering to the input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, features).
        weights : torch.Tensor
            Weight tensor of shape (channels, features).

        Returns
        -------
        torch.Tensor
            Locally filtered tensor.
        """
        weights = weights.unsqueeze(0).expand(x.size(0), -1, -1)

        x = self.activation_2(torch.mul(x, weights) - self.local_filter_bias)
        return x

    def _compute_adj(self, x, self_loop=True, epsilon=1e-6):
        """Computes the adjacency matrix for the GCN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_nodes, features).
        self_loop : bool, optional
            Whether to include self-loops in the adjacency matrix. Default: True.
        epsilon: float, optional, default 1E6
            Epislon to avoid log of zero

        Returns
        -------
        torch.Tensor
            Normalized adjacency matrix of shape (batch_size, num_nodes, num_nodes).
        """
        # Compute self-similarity
        x_trans = x.transpose(1, 2)
        s = torch.bmm(x, x_trans)  # Shape: (batch_size, num_nodes, num_nodes)

        global_adj_trans = self.global_adj.transpose(0, 1)
        adj = self.activation_2(s * (self.global_adj + global_adj_trans))

        if self_loop:
            adj += torch.eye(adj.size(1), device=adj.device).unsqueeze(0)

        # Normalize adjacency matrix
        rowsum = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(rowsum + epsilon, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj_normalized = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj_normalized


x = torch.zeros(1, 1, 32, 1000)

original_order = [
    "Fp1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "Oz",
    "Pz",
    "Fp2",
    "AF4",
    "Fz",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "Cz",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]

graph_idx = [
    ["Fp1", "Fp2"],
    ["AF3", "AF4"],
    ["F3", "F7", "Fz", "F4", "F8"],
    ["FC5", "FC1", "FC6", "FC2"],
    ["C3", "Cz", "C4"],
    ["CP5", "CP1", "CP2", "CP6"],
    ["P7", "P3", "Pz", "P4", "P8"],
    ["PO3", "PO4"],
    ["O1", "Oz", "O2"],
    ["T7"],
    ["T8"],
]


def get_channel_indices(
    original_order: List[str], graph_idx: List[List[str]]
) -> Tuple[List[int], List[int]]:
    """
    Generates indices of channels based on the provided graph definitions.

    Args:
        original_order (List[str]): The original ordering of channel names.
        graph_idx (List[List[str]]): A list of subgraphs, each containing channel names.

    Returns:
        Tuple[List[int], List[int]]:
            - idx: A flat list of indices corresponding to channel positions in original_order.
            - num_chan_local_graph: A list containing the number of channels in each subgraph.

    Raises:
        ValueError: If a channel in graph_idx is not found in original_order.
    """
    # Create a mapping from channel name to its index for O(1) lookups
    channel_to_index = {channel: idx for idx, channel in enumerate(original_order)}

    idx: List[int] = []
    num_chan_local_graph: List[int] = []

    for subgraph in graph_idx:
        subgraph_length = len(subgraph)
        num_chan_local_graph.append(subgraph_length)
        for chan in subgraph:
            if chan not in channel_to_index:
                raise ValueError(f"Channel '{chan}' not found in original_order.")
            idx.append(channel_to_index[chan])

    return idx, num_chan_local_graph


improved_idx, improved_num_chan = get_channel_indices(original_order, graph_idx)

LGGNet(
    n_chans=32,
    n_times=1000,
    n_outputs=2,
    sfreq=128,
    num_T=64,  # num_T controls the number of temporal filters
    out_graph=32,
    pool_size=16,
    pool_step_rate=0.25,
    idx_graph=improved_idx,
    dropout_rate=0.5,
)

with torch.no_grad():
    out = LGGNet(x)
