import sys 
import os 
import mlflow
from omegaconf import OmegaConf, DictConfig, open_dict
import numpy as np
import torch
from pathlib import Path
from source.models import MLP
from source.factories import  dataset_factory, model_factory
from source.factories import optimizers_factory, logger_factory, lr_schedulers_factory, trainer_factory
from datetime import datetime
from utils import flatten_dict
import hydra





with hydra.initialize(config_path="source/conf", version_base=None):  # version_base ensures compatibility; adjust if needed
    # Compose the config, specifying config_name here
    cfg: DictConfig = hydra.compose(config_name="config_mlp", overrides=["model=mlp"]) 
with open_dict(cfg):
    cfg.unique_id = f"{cfg.model.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    #cfg.dataset.node_sz = 200 
    cfg.run_dir = str(Path(cfg.log_path) / cfg.unique_id)

mlflow.set_experiment(cfg.get("mlp", "MLPExperiment"))
with mlflow.start_run(run_name=cfg.unique_id) as run:

    logger = logger_factory(cfg)
    logger.info(f"Starting MLflow Run: {run.info.run_id}")
    flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    mlflow.log_params(flat_cfg)
    logger.info("Logged configuration parameters to MLflow.")

    dataloaders = dataset_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(cfg, model=model)
    lr_schedulers = lr_schedulers_factory(cfg=cfg)
    trainer = trainer_factory(cfg, model, optimizers, lr_schedulers, dataloaders,logger)

    trainer.run()
    logger.info("MLflow run finished.")


