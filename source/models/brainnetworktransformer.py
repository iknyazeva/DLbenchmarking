import torch
import torch.nn as nn
from source.utils.bnt_components import TransPoolingEncoder
from omegaconf import DictConfig
from .base import BaseModel
from source.factories import positional_encoding_factory


class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        #self.pos_encoding = config.model.pos_encoding
        self.pos_encoder = positional_encoding_factory(config)
        #if self.pos_encoding == 'identity':
        #    self.node_identity = nn.Parameter(torch.zeros(
        #        config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
        #    forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
        #    nn.init.kaiming_normal_(self.node_identity)

        if self.pos_encoder is not None:
            # We can get the embedding dimension from the config
            forward_dim += config.model.pos_encoding.embed_dim

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

        if self.pos_encoder is not None:
            pos_emb = self.pos_encoder(node_feature)
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



