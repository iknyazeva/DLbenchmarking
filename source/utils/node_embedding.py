import torch
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
from source.utils import DEC
from omegaconf import DictConfig



class BasePositionalEncoding(nn.Module):
    """A base class for all positional encoding modules."""
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def forward(self, node_feature: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
class IdentityEncoding(BasePositionalEncoding):
    """Learnable node embeddings."""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.node_identity = nn.Parameter(torch.zeros(
            config.dataset.node_sz, config.model.pos_encoding.embed_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.node_identity)

    def forward(self, node_feature: torch.Tensor) -> torch.Tensor:
        bz = node_feature.shape[0]
        return self.node_identity.expand(bz, *self.node_identity.shape)
    
class RRWPEncoding(BasePositionalEncoding):
    """Computes Random Walk with Restart Positional Encodings on-the-fly."""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.walk_length = config.model.pos_encoding.embed_dim

    def forward(self, node_feature: torch.Tensor) -> torch.Tensor:
        # We only need the first matrix in the batch for the calculation
        # assuming the batch has the same graph structure.
        # If not, the add_full_rrwp would be used.
        return add_full_rrwp(node_feature, self.walk_length)



def add_full_rrwp(data, walk_length):
    pes = []
    for ids in range(data.shape[0]):
        dt = data[ids].squeeze()
        pe = add_every_rrwp(dt, walk_length)
        pes.append(pe)
    return torch.stack(pes)

def add_every_rrwp(data,
                  walk_length=8,
                  add_identity=True
                  ):

    edge_index = torch.column_stack(torch.where(data > 0.3)).T.contiguous()

    device = edge_index.device
    num_nodes = data.shape[0]
    edge_weight = data[edge_index[0], edge_index[1]]
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    adj = adj.to_dense()
    # Compute D^{-1} A:
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=device))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k


    return abs_pe