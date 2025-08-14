import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from omegaconf import DictConfig
from .base import BaseModel


class GraphTransformer(BaseModel):

    def __init__(self, cfg: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        self.readout = cfg.model.readout
        self.node_num = cfg.dataset.node_sz
        self.feature_dim = cfg.model.embedding_size if hasattr(cfg.model, 'embedding_size') else cfg.dataset.node_feature_sz 

        # Optional input projection
        if hasattr(cfg.model, 'embedding_size') and cfg.model.embedding_size != cfg.dataset.node_feature_sz:
            self.input_projection = nn.Linear(cfg.dataset.node_feature_sz, self.feature_dim)
        else:
            self.input_projection = nn.Identity()

        for _ in range(cfg.model.self_attention_layer):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=self.feature_dim, nhead=cfg.model.nhead, dropout=0.1,
                                        dim_feedforward=cfg.model.dim_feedforward,batch_first=True)
            )

        if cfg.model.embedding_size:
            final_dim = cfg.model.embedding_size
        else:
            final_dim = cfg.dataset.node_feature_sz

        if self.readout == "concat":
            self.dim_reduction = nn.Sequential(
                nn.Linear(final_dim, cfg.model.dim_reduction),
                nn.LeakyReLU()
            )
            final_dim = cfg.model.dim_reduction * self.node_num

        elif self.readout == "sum":
            self.norm = nn.BatchNorm1d(final_dim)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, node_feature):
        bz, _, _, = node_feature.shape
        node_feature = self.input_projection(node_feature)

        for atten in self.attention_list:
            node_feature = atten(node_feature)

        if self.readout == "concat":
            node_feature = self.dim_reduction(node_feature)
            node_feature = node_feature.reshape((bz, -1))

        elif self.readout == "mean":
            node_feature = torch.mean(node_feature, dim=1)
        elif self.readout == "max":
            node_feature, _ = torch.max(node_feature, dim=1)
        elif self.readout == "sum":
            node_feature = torch.sum(node_feature, dim=1)
            node_feature = self.norm(node_feature)

        return self.fc(node_feature)


