import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from .base_trainer import BaseTrainer

class SupervisedTrainer(BaseTrainer):
    """
    A concrete trainer for standard supervised classification tasks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Loss function specific to this training type
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def train_step(self, batch):
        # 1. Preprocess the batch using the unified method from the base class
        data_context = self._preprocess_batch(batch, 'train')
        
        # 2. Prepare model inputs from the context based on config
        model_args = [data_context[key] for key in self.cfg.model.forward_inputs]
        
        # 3. Forward pass
        logits = self.model(*model_args)
        
        # 4. Calculate loss
        # Note: The label in the context may be a soft label from mixup
        loss = self.criterion(logits, data_context['label'].long())
        
        # 5. Backward pass and optimizer step
        for opt in self.optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in self.optimizers:
            opt.step()
            
        # 6. Update custom LR schedulers (if used)
        for scheduler, optimizer in zip(self.schedulers, self.optimizers):
            if hasattr(scheduler, 'update'): # Check if it's our custom scheduler
                scheduler.update(optimizer, self.current_step)
        
        # 7. Return metrics for logging
        acc = (logits.argmax(dim=-1) == data_context['label']).float().mean()
        return {'loss': loss.item(), 'accuracy': acc.item()}

    def validation_step(self, batch):
        # Validation is similar but without backpropagation
        data_context = self._preprocess_batch(batch, 'validation')
        model_args = [data_context[key] for key in self.cfg.model.forward_inputs]
        logits = self.model(*model_args)
        
        # Calculate validation loss and metrics
        val_loss = self.criterion(logits, data_context['label'])
        val_acc = (logits.argmax(dim=-1) == data_context['label']).float().mean()
        
        # Calculate AUC
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = data_context['label'].cpu().numpy()
        try:
            val_auc = roc_auc_score(labels, probs)
        except ValueError:
            val_auc = 0.5 # Handle case with only one class in batch

        return {'val_loss': val_loss.item(), 'val_accuracy': val_acc.item(), 'val_auc': val_auc}
    
    def test_step(self, batch):
        """
        For the test phase, this step only performs the forward pass and returns
        the raw logits and labels for later aggregation.
        """
        data_context = self._preprocess_batch(batch, 'test')

        model_args = [data_context[key] for key in self.cfg.model.forward_inputs]
        logits = self.model(*model_args)
        
        # Return the raw tensors, they will be collected by the BaseTrainer
        return {
            'logits': logits,
            'labels': data_context['label']
        }
    
    @staticmethod
    def soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cross-entropy loss for soft labels.

        Args:
        logits: The raw model outputs (batch_size, num_classes).
        soft_labels: A FloatTensor of shape (batch_size, num_classes) where each
                     row sums to 1. This comes from the Mixup operation.

        Returns:
            The mean cross-entropy loss.
        """
        # First, apply log_softmax to the logits to get log-probabilities
        log_probs = F.log_softmax(logits, dim=1)
    
        # The loss is the negative sum of the product of soft labels and log-probabilities,
        # averaged over the batch.
        return -torch.sum(soft_labels * log_probs, dim=1).mean()