import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from .base import BaseModel

class BrainNetCNN(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.in_planes = 1

        self.d = config.dataset.node_sz
        self.p = config.model.dropout_rate

        self.e2e_layers = nn.ModuleList([])
        in_ch, out_ch = 1, config.model.E2E_channels
        for i in range(config.model.num_E2Eblock):
            self.e2e_layers.append(
                E2EBlock(in_ch, out_ch[i], self.d, bias=True))
            in_ch = out_ch[i]

        e2n_channel = config.model.E2N_channels
        self.E2N = nn.Conv2d(in_ch, e2n_channel, (1, self.d))

        n2g_channel = config.model.N2G_channels
        self.N2G = nn.Conv2d(e2n_channel, n2g_channel, (self.d, 1))

        self.classifier = nn.ModuleList([
            nn.LazyLinear(h) for h in config.model.hidden_out_Linear])


    def forward(self, node_feature: torch.tensor):
        node_feature = node_feature.unsqueeze(dim=1)

        for layer in self.e2e_layers:
            node_feature = F.leaky_relu(layer(node_feature), negative_slope=0.33)

        out = F.leaky_relu(
            self.E2N(node_feature), negative_slope=0.33)

        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=self.p)
        
        out = out.view(out.size(0), -1)

        for layer in self.classifier:
            out = F.leaky_relu(layer(out), negative_slope=0.33)

        return out
    
class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)    
