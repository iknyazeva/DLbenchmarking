import torch
import numpy as np
import logging
import mlflow
import pandas as pd
from abc import ABC, abstractmethod
from importlib import import_module
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from typing import List, Dict
from source.utils import EarlyStopping

class BaseTrainer(ABC):
    """
    An abstract base class for all training logic.
    It handles the generic parts: epoch looping, validation, logging, and saving.
    Subclasses must implement the specific logic for a training step.
    """
    def __init__(self, 
                 cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 schedulers: List[torch.optim.lr_scheduler._LRScheduler],
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 logger: logging.Logger):
        self.cfg = cfg
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.dataloaders = dataloaders
        self.logger = logger
        self.device = next(model.parameters()).device
        
        self.epochs = cfg.training.epochs
        self.current_step = 0
        self.current_epoch = 0

        if not hasattr(cfg, 'run_dir'):
            raise ValueError("Configuration object must have a 'run_dir' attribute.")
            
        self.run_dir = Path(cfg.run_dir)
        self.logger.info(f"Trainer will save artifacts to: {self.run_dir}")

        if 'patience' in cfg.training:
            self.logger.info(f"Early stopping enabled with patience: {cfg.training.patience}")
            self.early_stopping = EarlyStopping(
                patience=cfg.training.patience,
                verbose=True,
                path=self.run_dir / "best_model.pt", # It will handle saving the best model
                trace_func=self.logger.info
            )
        else:
            self.logger.info("Early stopping is disabled.")
            self.early_stopping = None
        
        # MLflow and artifact saving setup
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(exist_ok=True, parents=True)

    def _preprocess_batch(self, batch: tuple, mode: str) -> Dict[str, torch.Tensor]:
        """
        Handles the declarative preprocessing pipeline from the config.
        Dynamically loads and executes functions from the preprocess module.
        """
        # 1. Initialize the Data Context from the batch
        time_series, node_feature, label = batch
        data_context = {
            "time_series": time_series.to(self.device),
            "node_feature": node_feature.to(self.device),
            "label": label.to(self.device)
        }

        # 2. Get a reference to the preprocess module
        # This assumes your functions are in a file named `source/preprocess.py`
        try:
            preprocess_module = import_module("source.utils.prepossess")
        except ImportError:
            self.logger.warning("Could not find 'source.utils.prepossess.py'. Skipping preprocessing.")
            return data_context

        # 3. Execute the declarative pipeline from the config
        # Use .get('pipeline', []) to safely handle cases where the pipeline is not defined
        if mode == 'train':
            pipeline = self.cfg.preprocess.get('train_pipeline', [])
        else: # For 'validation' or 'test' modes
            pipeline = self.cfg.preprocess.get('eval_pipeline', [])

        for step in pipeline:
            function_name = step['function']
            
            # Get the function from the preprocess module
            try:
                func_to_run = getattr(preprocess_module, function_name)
            except AttributeError:
                self.logger.error(f"Function '{function_name}' not found in 'source/preprocess.py'. Skipping step.")
                continue

            # Gather input arguments for the function from the current data_context
            try:
                args = [data_context[key] for key in step['inputs']]
            except KeyError as e:
                self.logger.error(f"Input key {e} for function '{function_name}' not found in data context. Skipping step.")
                continue

            # Execute the function
            results = func_to_run(*args)
            
            # Ensure results are a tuple for consistent unpacking
            if not isinstance(results, tuple):
                results = (results,)
            
            # Check if the number of outputs matches the config
            if len(results) != len(step['outputs']):
                self.logger.error(f"Mismatch between function '{function_name}' output count ({len(results)}) and config output count ({len(step['outputs'])}). Skipping update.")
                continue

            # Update the data_context with the function's outputs
            for i, key in enumerate(step['outputs']):
                data_context[key] = results[i]
        
        return data_context

    @abstractmethod
    def train_step(self, batch: tuple) -> Dict[str, float]:
        """
        Perform a single training step on a batch of data.
        Must be implemented by subclasses.
        Should return a dictionary of metrics for logging (e.g., {'loss': 0.5}).
        """
        pass

    @abstractmethod
    def validation_step(self, batch: tuple) -> Dict[str, float]:
        """
        Perform a single validation step on a batch of data.
        Must be implemented by subclasses.
        Should return a dictionary of metrics for logging (e.g., {'val_loss': 0.4}).
        """
        pass

    def _run_epoch(self, mode: str) -> pd.DataFrame:
        """Generic method to run a full epoch for either training or validation."""
        if mode == 'train':
            self.model.train()
            dataloader = self.dataloaders['train']
            step_func = self.train_step
        elif mode == 'validation':
            self.model.eval()
            dataloader = self.dataloaders['validation']
            step_func = self.validation_step
        elif mode == 'test':
            self.model.eval()
            dataloader = self.dataloaders['test']
            step_func = self.test_step
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if mode in ['train', 'validation']:
            epoch_metrics = []
            with torch.set_grad_enabled(mode == 'train'):
                for batch in dataloader:
                    if mode == 'train':
                        self.current_step += 1
                        self.on_train_step_start() 
                    step_metrics = step_func(batch)
                    epoch_metrics.append(step_metrics)
            return pd.DataFrame(epoch_metrics).mean().to_dict()
        elif mode == 'test':
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for batch in dataloader:
                    step_output = step_func(batch) # This returns {'logits': ..., 'labels': ...}
                    all_logits.append(step_output['logits'])
                    all_labels.append(step_output['labels'])
            
            # Concatenate all batches into single tensors
            final_logits = torch.cat(all_logits, dim=0)
            final_labels = torch.cat(all_labels, dim=0)
            return final_logits, final_labels
    
    def run(self):
        """The main training loop."""
        self.logger.info("Starting training...")

        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            self.current_epoch = epoch
            
            # Run training epoch
            train_metrics = self._run_epoch('train')

            val_metrics = self._run_epoch('validation')
            
            # Combine and log metrics
            combined_metrics = {**train_metrics, **val_metrics}
            mlflow.log_metrics(combined_metrics, step=self.current_epoch)
            
            # Log to console
            log_str = f"Epoch {self.current_epoch}: " + " | ".join([f"{k}: {v:.4f}" for k, v in combined_metrics.items()])
            self.logger.info(log_str)
            
            
            current_val_loss = combined_metrics.get('val_loss', np.inf)
            self.on_validation_epoch_end(current_val_loss)

            if self.early_stopping:
                self.early_stopping(current_val_loss, self.model)
                if self.early_stopping.early_stop:
                    self.logger.info("Early stopping triggered. Halting training.")
                    break  # Exit the training loop
            else:
                best_val_loss = getattr(self, 'best_val_loss', np.inf)

                if current_val_loss < best_val_loss:
                    self.logger.info(f"New best validation loss: {current_val_loss:.4f}. Saving model.")
                    self.best_val_loss = current_val_loss
                    torch.save(self.model.state_dict(), self.run_dir / "best_model.pt")
                    mlflow.log_metric("best_val_loss", best_val_loss, step=self.current_epoch)
        
        final_epoch = self.current_epoch
        if self.early_stopping and self.early_stopping.early_stop:
            # If early stopping was the reason we stopped, log it as a parameter/tag.
            # A "tag" is often better for this kind of metadata.
            mlflow.set_tag("status", "EARLY_STOPPED")
            mlflow.log_param("stopped_epoch", final_epoch)
            self.logger.info("Logged early stopping status to MLflow.")
        else:
            # If the training ran to completion.
            mlflow.set_tag("status", "COMPLETED")
            self.logger.info("Logged completed status to MLflow.")
        self.logger.info("Training finished.")
        self.test()

    def test(self):
        """
        Loads the best model and evaluates it on the test set.
        """
        self.logger.info("--- Starting Testing Phase ---")
        
        best_model_path = self.run_dir / "best_model.pt"
        if not best_model_path.exists():
            self.logger.warning("No best model found to test. Using the final model instead.")
        else:
            self.logger.info(f"Loading best model from: {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        all_logits, all_labels = self._run_epoch('test')
        test_metrics = self._calculate_test_metrics(all_logits, all_labels)
        
        self.logger.info("--- Test Results ---")
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
        self.logger.info(log_str)
        
        final_test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        log_files = list(self.run_dir.glob("*.log"))
        mlflow.log_artifact(str(log_files[0]))
        mlflow.log_metrics(final_test_metrics)
        if self.cfg.training.get('log_artifacts', False):
            self.logger.info("Artifact logging is enabled. Logging model and log files...")
            
            # Log the saved best model file
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), artifact_path="model")
            
            
            self.logger.info("Finished logging artifacts.")
        else:
            self.logger.info("Artifact logging is disabled.")


    def _calculate_test_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculates all required metrics from the complete test set outputs.
        """
        # Convert tensors to numpy arrays on the CPU
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
        # Ensure labels are 1D class indices
        if labels.ndim == 2:
            labels = labels.argmax(dim=1)
        labels = labels.cpu().numpy()

        # --- Re-implementing the logic from your original code ---
        
        # 1. Calculate AUC
        auc = roc_auc_score(labels, probs)
        
        # 2. Get binary predictions by thresholding probabilities at 0.5
        preds_binary = (probs > 0.5).astype(int)
        acc = accuracy_score(labels, preds_binary)
        
        # 3. Calculate precision, recall, f1-score (micro-averaged)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds_binary, average='micro'
        )

        # 4. Get the full classification report to extract per-class recall (sensitivity/specificity)
        report = classification_report(labels, preds_binary, output_dict=True, zero_division=0)
        
        # Specificity (Recall of class 0) and Sensitivity (Recall of class 1)
        specificity = report.get('0', {}).get('recall', 0.0)
        sensitivity = report.get('1', {}).get('recall', 0.0)
        
        # 5. Assemble the final metrics dictionary
        metrics_dict = {
            'auc': auc,
            'acc': acc,
            'precision_micro': precision,
            'recall_micro': recall,
            'f1_micro': f1,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        return metrics_dict
    
    def on_train_step_start(self):
        """Hook called at the beginning of each training step."""
        # Update custom LR schedulers that work on a per-step basis
        for scheduler, optimizer in zip(self.schedulers, self.optimizers):
            if scheduler is not None and hasattr(scheduler, 'update'):
                scheduler.update(optimizer, self.current_step)

    def on_validation_epoch_end(self, val_loss: float):
        """Hook called after a validation epoch. Used for standard schedulers."""
        # This is where you would put logic for schedulers like ReduceLROnPlateau
        for scheduler in self.schedulers:
            if scheduler is not None and not hasattr(scheduler, 'update'):
                # This ensures we only step schedulers that are not our custom per-step one
                scheduler.step(val_loss)

