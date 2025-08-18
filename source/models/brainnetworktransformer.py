import torch
from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
from utils import DEC
from omegaconf import DictConfig
from .base import BaseModel


class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=config.model.dim_feedforward,
                                    output_node_num=size,
                                    nhead=config.model.nhead,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, config.model.dim_reduction),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(config.model.dim_reduction * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, node_feature: torch.tensor):

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all



class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, nhead=4,
                 pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=nhead,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)
       
       # self.transformer = TransformerEncoderLayer(d_model=input_feature_size, nhead=4,
       #                                             dim_feedforward=hidden_size,
       #                                             dropout=0.1,
       #                                             batch_first=True)
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)



class InterpretableTransformerEncoder(nn.TransformerEncoderLayer):
    """
    A TransformerEncoderLayer that saves the attention weights from its forward pass.
    """
    def __init__(self, *args, **kwargs):
        # The __init__ method doesn't need to change.
        super().__init__(*args, **kwargs)
        self.attention_weights: Optional[Tensor] = None

    # --- THIS IS THE CHANGE ---
    # We override the public `forward` method directly.
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        # This part of the logic is adapted from the source code of nn.TransformerEncoderLayer
        # to ensure compatibility and capture the weights correctly.
        
        # --- Self-Attention Block ---
        # We call the self-attention layer and explicitly ask for the weights.
        if self.norm_first:
            # Pre-normalization (LayerNorm -> Attention -> Add -> LayerNorm -> FF -> Add)
            sa_output, self.attention_weights = self.self_attn(
                self.norm1(src), src, src, # Pass normalized src to attention
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True
            )
            x = src + self.dropout1(sa_output)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-normalization (Attention -> Add -> LayerNorm -> FF -> Add -> LayerNorm)
            sa_output, self.attention_weights = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True
            )
            x = self.norm1(src + self.dropout1(sa_output))
            x = self.norm2(x + self._ff_block(x))
        
        return x

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights