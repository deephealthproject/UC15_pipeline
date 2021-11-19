"""
Module with utilities for the training pipeline.
"""
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics import functional as Fmetrics
import pytorch_lightning.callbacks as callbacks


class FeatureExtractorFreezeUnfreeze(callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=1):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # Freeze the feature extractor module
        print("Going to freeze the feature extractor")
        # self.freeze(pl_module.feature_extractor)  # Not working properly
        for param in pl_module.model.feature_extractor.parameters():
            param.requires_grad = False

    def finetune_function(self, pl_module, current_epoch,
                          optimizer, optimizer_idx):
        # If we reached the desired epoch, unfreeze the feature extractor
        if current_epoch == self.unfreeze_at_epoch:
            print("Going to unfreeze the feature extractor")
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.feature_extractor,
                optimizer=optimizer,
                train_bn=True,
                initial_denom_lr=1.0
            )


class ImageClassifier(pl.LightningModule):
    """
    Wrapper to train an image classifier model.
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: str = "Adam",
                 learning_rate: float = 0.0001,
                 l2_penalty: float = 0.0):
        """
        Model constructor.

        Args:
            model: Pytorch module of the model to train.

            optimizer: Optimizer type to use (choices: ["Adam", "SGD"]).

            learning_rate: Learning rate to use in the optimizer.

            l2_penalty: Value tu use for the weight decay in the optimizer.
        """
        super().__init__()

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.pretrained = model.pretrained  # We need this for the pipeline
        self.l2_penalty = l2_penalty

        self.metric = Accuracy()

        self.last_val_loss = None
        self.last_val_acc = None

        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)  # From one hot to indexes

        logits = self(x)  # Forward

        # Compute loss and metrics
        loss = F.cross_entropy(logits, y)
        acc = self.metric(logits, y)

        self.log('batch_loss', loss, prog_bar=True, logger=False)
        self.log('batch_acc', acc, prog_bar=True, logger=False)

        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        train_loss = mean(map(lambda x: x['loss'].item(), outputs))
        train_acc = mean(map(lambda x: x['acc'].item(), outputs))

        # Store the final epoch results with the logger
        self.log('train_loss', train_loss, logger=True)
        self.log('train_acc', train_acc, logger=True)
        if self.last_val_loss is not None:
            self.log('val_loss', self.last_val_loss, logger=True)
            self.log('val_acc', self.last_val_acc, logger=True)
        else:
            raise Exception("Validation metrics not found!")

        print(('\nEpoch results: '
               f'train_loss={train_loss:.4f} - '
               f'train_acc={train_acc:.4f} - '
               f'val_loss={self.last_val_loss:.4f} - '
               f'val_acc={self.last_val_acc:.4f}\n'))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)  # From one hot to indexes

        logits = self(x)  # Forward

        # Compute loss and metrics
        loss = F.cross_entropy(logits, y)
        acc = self.metric(logits, y)

        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        # Store the values to log them at the end of the training epoch
        self.last_val_loss = mean(map(lambda x: x['loss'].item(), outputs))
        self.last_val_acc = mean(map(lambda x: x['acc'].item(), outputs))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)  # From one hot to indexes

        logits = self(x)  # Forward

        # Compute loss and metrics
        loss = F.cross_entropy(logits, y)
        acc = self.metric(logits, y)

        self.log('test_loss', loss, prog_bar=False, on_epoch=True, logger=True)
        self.log('test_acc', acc, prog_bar=False, on_epoch=True, logger=True)

        return {'loss': loss, 'acc': acc, 'y_pred': logits, 'y_true': y}

    def test_epoch_end(self, outputs):
        preds = torch.cat([out['y_pred'] for out in outputs])
        targets = torch.cat([out['y_true'] for out in outputs])

        # Compute test metrics
        precision = Fmetrics.precision(preds, targets,
                                       num_classes=preds.shape[1])
        recall = Fmetrics.recall(preds, targets,
                                 num_classes=preds.shape[1])
        conf_matrix = Fmetrics.confusion_matrix(preds, targets,
                                                num_classes=preds.shape[1])

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print("Confusion matrix:\n", conf_matrix)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            return Adam(filter(lambda p: p.requires_grad, self.parameters()),
                        lr=self.learning_rate,
                        weight_decay=self.l2_penalty)
        if self.optimizer == "SGD":
            return SGD(filter(lambda p: p.requires_grad, self.parameters()),
                       lr=self.learning_rate,
                       momentum=0.9,
                       weight_decay=self.l2_penalty)

        raise Exception("Wrong optimizer name provided!")
