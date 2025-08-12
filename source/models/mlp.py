import torch
import torch.nn as nn
from omegaconf import DictConfig
from .base import BaseModel
import numpy as np

class MLP(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        node_sz = cfg.dataset.node_sz
        # The input size is the number of elements in the upper triangle of the correlation matrix
        # The formula for this is N * (N - 1) / 2
        input_dim = int(node_sz * (node_sz - 1) / 2)

        hidden_layers = cfg.model.hidden_layers
        dropout_rate = cfg.model.dropout_rate
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        # Dynamically create hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.extend([
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_layers[i], hidden_layers[i+1])
            ])
        
        # Add activation and dropout for the last hidden layer
        layers.extend([
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        # Output layer
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_layers[-1], 2) # 2 classes for output

    def forward(self, corr_vector):
        # This model only needs the correlation vector
        hidden_state = self.network(corr_vector)
        out = self.fc(hidden_state)
        return out

    def get_attention_weights(self):
        # MLP does not have attention weights
        return None