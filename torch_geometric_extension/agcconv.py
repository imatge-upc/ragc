"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
import math


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class AGCConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='mean',
                 flow='target_to_source',
                 root_weight=False, bias=False):

        super(AGCConv, self).__init__(aggr=aggr, flow=flow)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._wg = nn
        self.aggr = aggr
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        weights = self._wg(edge_attr)
        weights = weights.view(-1, self.in_channels, self.out_channels)
        return self.propagate(edge_index, x=x, weights=weights)

    def message(self, x_j, edge_index_i, weights):

        return torch.matmul(x_j.unsqueeze(1), weights).squeeze(1)

    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        s = '{}({}, {}'.format(self.__class__.__name__, self.in_channels, self.out_channels)
        s += ', bias={}'.format(False if self.bias is None else True)
        return s.format(**self.__dict__)
