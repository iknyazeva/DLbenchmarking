from typing import List, Optional, Tuple
import logging
from pathlib import Path
from importlib import import_module
from omegaconf import DictConfig
from torch.utils import data as utils
import torch
import torch.nn as nn
# Import your data loading functions and models
from source.dataset.dataloader import init_stratified_dataloader, init_dataloader, init_group_dataloader # Assuming this is your stratified loader
from source.components import LRScheduler, get_param_group_no_wd
from source.components.logger import set_file_handler


def dataset_factory(cfg: DictConfig) -> List[torch.utils.data.DataLoader]:
    """
    Dynamically loads a dataset and creates dataloaders based on configuration.

    This factory expects:
    1. `cfg.dataset.name` to be the lowercase name of the dataset (e.g., "abide").
    2. A corresponding Python module in `source/data/` (e.g., `source/data/abide.py`).
    3. A function named `load_{dataset_name}_data` (e.g., `load_abide_data`) within
       that module, which returns a tuple of raw data tensors.
    """
    dataset_name = cfg.dataset.name
    
    # 1. Construct the module path and the function name from convention.
    module_path = f"source.dataset.{dataset_name}"
    function_name = f"load_{dataset_name}_data"

    # 2. Dynamically import the dataset module.
    try:
        dataset_module = import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_path}'. "
            f"Ensure a file named '{dataset_name}.py' exists in the 'source/data/' directory."
        ) from e

    # 3. Get the data-loading function from the imported module.
    try:
        load_function = getattr(dataset_module, function_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' does not have a function named '{function_name}'. "
            "Please ensure the data-loading function is defined and named correctly."
        ) from e

    # 4. Call the function to get the raw data tensors.
    # The result is expected to be a tuple, e.g., (timeseries, pearson, labels, site)
    raw_data_tuple = load_function(cfg)

    # 5. Create the final DataLoader objects based on the 'stratified' flag.
    # Use .get() to safely access the key, defaulting to False if it's not present.

    #if cfg.dataset.get('stratified', False):
    if cfg.dataset.stratified:
        logging.info(f"Using stratified dataloader for '{dataset_name}' dataset.")
        dataloaders = init_stratified_dataloader(cfg, *raw_data_tuple)

    elif cfg.dataset.groups:
        logging.info(f"Using group dataloader for '{dataset_name}' dataset.")
        dataloaders = init_group_dataloader(cfg, *raw_data_tuple)

    else:
        logging.info(f"Using standard dataloader for '{dataset_name}' dataset.")
        # You would need to have this non-stratified version implemented
        dataloaders = init_dataloader(cfg, *raw_data_tuple)
            
    return dataloaders

def model_factory(cfg: DictConfig) -> torch.nn.Module:
    """
    Dynamically creates a model instance from configuration.

    This factory expects:
    1. `cfg.model.name` to be the CamelCase name of the model class (e.g., "MLP").
    2. A corresponding python file in `source/models/` named after the model class
       in all lowercase (e.g., "mlp.py").
    """
    model_name_str = cfg.model.name
    
    # 1. Determine the module path from the model name.
    #    Convention: 'MLP' -> 'mlp', 'GraphTransformer' -> 'graphtransformer'
    module_name = model_name_str.lower()
    module_path = f"source.models.{module_name}"

    # 2. Dynamically import the module.
    try:
        model_module = import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_path}'. "
            f"Ensure there is a file named '{module_name}.py' in the 'source/models/' directory."
        ) from e

    # 3. Get the model class from the imported module.
    #    Convention: The class name must match `model_name_str` exactly.
    try:
        model_class = getattr(model_module, model_name_str)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' does not have a class named '{model_name_str}'. "
            "Please ensure the class name in the Python file matches the 'name' in your model's YAML config."
        ) from e

    # 4. Instantiate the class with the configuration and return it.
    model = model_class(cfg)
    logging.info(f'Will be used model class[{model_name_str}]')

    
    return model


def optimizer_factory(model: torch.nn.Module, optimizer_config: DictConfig) -> torch.optim.Optimizer:
    parameters = {
        'lr': 0.0,
        'weight_decay': optimizer_config.weight_decay
    }

    if optimizer_config.no_weight_decay:
        params, _ = get_param_group_no_wd(model,
                                          match_rule=optimizer_config.match_rule,
                                          except_rule=optimizer_config.except_rule)
    else:
        params = list(model.parameters())
        logging.info(f'Parameters [normal] length [{len(params)}]')

    parameters['params'] = params

    optimizer_type = optimizer_config.name
    if optimizer_type == 'SGD':
        parameters['momentum'] = optimizer_config.momentum
        parameters['nesterov'] = optimizer_config.nesterov
    return getattr(torch.optim, optimizer_type)(**parameters)


def optimizers_factory(cfg: DictConfig, model: torch.nn.Module) -> List[torch.optim.Optimizer]:
    """
    Manager function: Creates a LIST of optimizers from the configuration.

    It assumes `cfg.optimizer` is a list of optimizer configurations,
    which is automatically assembled by Hydra's defaults list.
    """
    if not hasattr(cfg, 'optimizer'):
        raise ValueError(
            "Configuration does not have an 'optimizer' key. "
            "Please ensure your main config.yaml includes a defaults list for the optimizer."
        )

    # Hydra's defaults list creates cfg.optimizer for us. We just iterate through it.
    optimizer_configs = cfg.optimizer
    
    logging.info(f"Creating {len(optimizer_configs)} optimizer(s) based on the configuration.")
    
    # Use the singular factory to build each optimizer in the list
    return [optimizer_factory(model=model, optimizer_config=single_config) for single_config in optimizer_configs]


def lr_schedulers_factory(cfg: DictConfig) -> List[Optional[LRScheduler]]:
    """
    Creates a list of custom LRScheduler instances from the configuration.

    It iterates through the list of optimizer configs found in `cfg.optimizer`
    and creates a scheduler for each one that has an `lr_scheduler` block defined.
    
    If an optimizer config does not have an `lr_scheduler` block, a `None` is
    placed in the list for that optimizer.
    """
    if not hasattr(cfg, 'optimizer'):
        raise ValueError("Configuration must have an 'optimizer' key.")
        
    optimizer_configs = cfg.optimizer
    schedulers = []

    logging.info(f"Creating schedulers for {len(optimizer_configs)} optimizer(s)...")

    for opt_cfg in optimizer_configs:
        # Check if a scheduler is defined for this specific optimizer
        if 'lr_scheduler' in opt_cfg:
            schedulers.append(LRScheduler(cfg=cfg, optimizer_cfg=opt_cfg))
        else:
            # If not defined, append None. This allows some optimizers
            # to have a scheduler while others don't.
            schedulers.append(None)
            
    return schedulers

def logger_factory(config: DictConfig) -> Tuple[logging.Logger]:
    log_path = Path(config.log_path) / config.unique_id
    log_path.mkdir(exist_ok=True, parents=True)
    log_file_path = log_path / "_run.log"
    logger = set_file_handler(log_file_path)
    logger.info(f"Run directory created at: {log_path}")
    return logger


def trainer_factory(cfg: DictConfig, model, optimizers, schedulers, dataloaders, logger):
    """Dynamically creates a trainer instance from configuration."""
    training_cfg = cfg.training
    class_name = training_cfg.class_name
    module_name = training_cfg.module_name
    
    module_path = f"source.training.{module_name}"
    try:
        trainer_module = import_module(module_path)
        trainer_class = getattr(trainer_module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not find trainer class '{class_name}' in module '{module_path}'. Check your training config.") from e
        
    # Instantiate the trainer with all the required components
    return trainer_class(
        cfg=cfg,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        dataloaders=dataloaders,
        logger=logger
    )

def positional_encoding_factory(cfg: DictConfig) -> nn.Module:
    """Dynamically creates a positional encoding module from config."""
    # If no pos_encoding is defined, return None
    if 'pos_encoding' not in cfg.model or cfg.model.pos_encoding is None:
        return None
    if cfg.model.pos_encoding.name.lower() == 'none':
        return None

    pe_cfg = cfg.model.pos_encoding
    class_name = pe_cfg.name
    
    # Dynamically import the class from our new components file
    try:
        pe_module = import_module("source.utils.node_embedding")
        pe_class = getattr(pe_module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not find PositionalEncoding class '{class_name}'.") from e
        
    return pe_class(cfg)

def components_factory(cfg: DictConfig, model: torch.nn.Module):
    """
    Factory to create optimizer, criterion, and scheduler.
    """
    # Criterion
    # Assuming CrossEntropyLoss for classification
    criterion = torch.nn.CrossEntropyLoss()
    
    # Optimizer
    opt_cfg = cfg.optimizer
    if opt_cfg.name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay
        )
    else:
        raise ValueError(f"Optimizer '{opt_cfg.name}' not supported.")

    # Scheduler (optional)
    scheduler = None # Can be expanded later
    if 'lr_scheduler' in opt_cfg:
        # Example for a simple scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return criterion, optimizer, scheduler
