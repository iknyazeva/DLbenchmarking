import torch
import numpy as np
import random
import logging

def set_seed(seed: int):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    if seed is not None:
        logging.info(f"Setting global seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The two lines below are for full reproducibility with CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False