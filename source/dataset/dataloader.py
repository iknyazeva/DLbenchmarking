import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
import numpy as np
import torch.nn.functional as F


def _update_config_with_steps(cfg: DictConfig, train_length: int):
    """Calculates and injects step information into the configuration."""
    with open_dict(cfg):
        cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

def _create_dataloaders_from_datasets(
    cfg: DictConfig,
    train_dataset: utils.Dataset,
    val_dataset: utils.Dataset,
    test_dataset: utils.Dataset
) -> List[utils.DataLoader]:
    """Creates the final DataLoader objects from split datasets."""
    train_dataloader = utils.DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        drop_last=cfg.dataset.get('drop_last', True)
    )
    val_dataloader = utils.DataLoader(
        val_dataset,
        batch_size=cfg.dataset.val_batch_size,
        shuffle=False
    )
    test_dataloader = utils.DataLoader(
        test_dataset,
        batch_size=cfg.dataset.test_batch_size,
        shuffle=False
    )
    return {
        'train': train_dataloader,
        'validation': val_dataloader,
        'test': test_dataloader
        }


def init_dataloader(cfg: DictConfig, *data_tuple) -> List[utils.DataLoader]:
    """
    Splits data randomly (non-stratified) and creates DataLoaders.
    """
    final_timeseires, final_pearson, labels, _ = data_tuple
    
    # The label tensor from the loader needs to be 1D for random_split
    labels = labels.long() 
    
    length = len(final_timeseires)
    train_length = int(length * cfg.dataset.train_set)
    val_length = int(length * cfg.dataset.val_set)
    test_length = length - train_length - val_length

    _update_config_with_steps(cfg, train_length)

    generator = torch.Generator().manual_seed(cfg.seed)
    full_dataset = utils.TensorDataset(final_timeseires, final_pearson, labels)
    train_dataset, val_dataset, test_dataset = utils.random_split(
        full_dataset, [train_length, val_length, test_length], generator=generator
    )
    
    return _create_dataloaders_from_datasets(cfg, train_dataset, val_dataset, test_dataset)

def init_stratified_dataloader(cfg: DictConfig, *data_tuple) -> List[utils.DataLoader]:
    """
    Splits data using stratification based on site and creates DataLoaders.
    """
    final_timeseires, final_pearson, labels, site_info, groups = data_tuple

    # The label tensor needs to be 1D for this function's logic
    labels = labels.long()

    length = len(final_timeseires)
    train_length = int(length * cfg.dataset.train_set)
    val_length = int(length * cfg.dataset.val_set)
    test_length = length - train_length - val_length

    _update_config_with_steps(cfg, train_length)

    # First split: Separate training set from (validation + test) set
    split1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_length + test_length), train_size=train_length, random_state=cfg.seed

    )
    train_idx, val_test_idx = next(split1.split(final_timeseires, site_info))
    
    # Second split: Separate validation and test sets from the remainder
    val_test_site_info = site_info[val_test_idx]
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=cfg.seed)
    val_idx_rel, test_idx_rel = next(split2.split(final_timeseires[val_test_idx], val_test_site_info))
    
    # Convert relative indices to absolute indices
    val_idx = val_test_idx[val_idx_rel]
    test_idx = val_test_idx[test_idx_rel]

    # Create datasets from the stratified indices
    train_dataset = utils.TensorDataset(final_timeseires[train_idx], final_pearson[train_idx], labels[train_idx])
    val_dataset = utils.TensorDataset(final_timeseires[val_idx], final_pearson[val_idx], labels[val_idx])
    test_dataset = utils.TensorDataset(final_timeseires[test_idx], final_pearson[test_idx], labels[test_idx])

    return _create_dataloaders_from_datasets(cfg, train_dataset, val_dataset, test_dataset)

def init_group_dataloader(cfg: DictConfig, *data_tuple):

    final_timeseires, final_pearson, labels, site_info, groups = data_tuple

    labels = labels.long()

    length = len(final_timeseires)
    train_length = int(length * cfg.dataset.train_set)
    val_length = int(length * cfg.dataset.val_set)
    test_length = length - train_length - val_length

    _update_config_with_steps(cfg, train_length)

    # First split: Separate training set from (validation + test) set
    split1 = GroupShuffleSplit(n_splits=1, 
                               #test_size=(val_length + test_length), 
                               train_size=cfg.dataset.train_set, 
                               random_state=cfg.seed)
    
    for train_idx, val_idx in split1.split(final_timeseires, groups, groups=groups):

        train_dataset = utils.TensorDataset(final_timeseires[train_idx], final_pearson[train_idx], labels[train_idx])
        val_dataset = utils.TensorDataset(final_timeseires[val_idx], final_pearson[val_idx], labels[val_idx])
        test_dataset = utils.TensorDataset(final_timeseires[val_idx], final_pearson[val_idx], labels[val_idx])
    
    return _create_dataloaders_from_datasets(cfg, train_dataset, val_dataset, test_dataset)
