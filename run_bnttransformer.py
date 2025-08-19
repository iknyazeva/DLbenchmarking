from omegaconf import OmegaConf, DictConfig, open_dict
import numpy as np
from source.utils import set_seed
import hydra
from source.training import model_training
import numpy as np
import torch
from pathlib import Path
from source.models import BrainNetworkTransformer



@hydra.main(version_base=None, config_path="source/conf", config_name="config_bnttransformer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    for _ in range(cfg.repeat_time):
        model_training(cfg)

if __name__ == '__main__':
    main()