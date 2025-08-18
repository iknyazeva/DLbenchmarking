from random import uniform, randint
import torch
from torch import nn
from omegaconf import DictConfig



class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class MeanMLP(nn.Module):
    """
    meanMLP model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        hidden_size: int
        num_layers: int
    Data info expected in model_cfg:
        input_size: int - input_feature_size
        output_size: int - n_classes
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        dropout = cfg.model.dropout_rate
        hidden_size = cfg.model.hidden_size
        num_layers = cfg.model.num_layers
        output_size = cfg.model.output_size

        #if len(hidden_size) != 0:

        # input block
        layers = [
            nn.LazyLinear(hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]
        # inter blocks: default HPs model has none of them
        for i in range(num_layers):
            layers.append(
                nn.Sequential(
                    ResidualBlock(
                        nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                        )
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ),
            )

        # output block
        layers.append(
            nn.Linear(hidden_size, output_size),
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, introspection=False):
        #print("!!!!!!!!!!!!!!!!!!!!1x.shape", x.shape)
        x = x.permute(0, 2, 1)
        bs, tl, fs = x.shape  # [batch_size, time_length, input_feature_size]

        # view: Теперь каждый временной шаг в каждом объекте батча 
        # становится отдельным "экземпляром" для пропуска через слои сети.
        fc_output = self.fc(x.reshape(-1, fs))
        fc_output = fc_output.view(bs, tl, -1)

        logits = fc_output.mean(1)

        if introspection:
            predictions = torch.argmax(logits, dim=-1)
            return fc_output, predictions

        return logits