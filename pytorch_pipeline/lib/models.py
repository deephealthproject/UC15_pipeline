"""
Module to create the models architectures.
"""
import argparse
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics import functional as Fmetrics
import torchvision.models as models


class ResNet(pl.LightningModule):
    """
    Class to create a model using transfer learning from a ResNet model.
    """

    def __init__(self,
                 resnet_version: int,
                 num_classes: int,
                 optimizer: str = "Adam",
                 learning_rate: float = 0.0001,
                 pretrained: bool = True,
                 l2_penalty: float = 0.0):
        """
        Model constructor.

        Args:
            resnet_version: Choices: [18, 34, 50, 101, 152]

            num_classes: Number of units to put in the last Dense layer.

            optimizer: Optimizer type to use (choices: ["Adam", "SGD"]).

            learning_rate: Learning rate to use in the optimizer.

            pretrained: To use or not the pretrained weights from imagenet

            l2_penalty: Value tu use for the weight decay in the optimizer.
        """
        super().__init__()

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.pretrained = pretrained
        self.l2_penalty = l2_penalty

        self.metric = Accuracy()

        self.last_val_loss = None
        self.last_val_acc = None

        if resnet_version == 18:
            backbone = models.resnet18(pretrained=self.pretrained)
        elif resnet_version == 34:
            backbone = models.resnet34(pretrained=self.pretrained)
        elif resnet_version == 50:
            backbone = models.resnet50(pretrained=self.pretrained)
        elif resnet_version == 101:
            backbone = models.resnet101(pretrained=self.pretrained)
        elif resnet_version == 152:
            backbone = models.resnet152(pretrained=self.pretrained)

        # Extract the pretrained convolutional part
        layers = list(backbone.children())[:-1]  # Get the convolutional part
        self.feature_extractor = nn.Sequential(*layers)

        # Prepare the new classifier
        num_filters = backbone.fc.in_features
        self.fully_connected = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(num_filters // 2, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        out = self.fully_connected(x)
        return out

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


def get_model(model_name: str,
              in_shape: tuple,
              num_classes: int,
              args: argparse.Namespace) -> pl.LightningModule:
    """
    Auxiliary function to create the selected model topology.

    Args:
        model_name: Name of the architecture of the model to create.

        in_shape: Tuple with the input shape of the model (without batch dim).

        num_classes: Number of units in the last layer for classification.

        args: Aditional arguments provided by the argparse in the main script.

    Returns:
        The Pytorch Lightning module of the selected model.
    """
    # NO pretrained ResNet models
    if model_name == "ResNet18":
        return ResNet(18,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      pretrained=False,
                      l2_penalty=args.l2_penalty)
    if model_name == "ResNet34":
        return ResNet(34,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      pretrained=False,
                      l2_penalty=args.l2_penalty)
    if model_name == "ResNet50":
        return ResNet(50,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      pretrained=False,
                      l2_penalty=args.l2_penalty)
    if model_name == "ResNet101":
        return ResNet(101,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      pretrained=False,
                      l2_penalty=args.l2_penalty)
    if model_name == "ResNet152":
        return ResNet(152,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      pretrained=False,
                      l2_penalty=args.l2_penalty)

    # Pretrained ResNet models
    if model_name == "PretrainedResNet18":
        return ResNet(18,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      l2_penalty=args.l2_penalty)
    if model_name == "PretrainedResNet34":
        return ResNet(34,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      l2_penalty=args.l2_penalty)
    if model_name == "PretrainedResNet50":
        return ResNet(50,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      l2_penalty=args.l2_penalty)
    if model_name == "PretrainedResNet101":
        return ResNet(101,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      l2_penalty=args.l2_penalty)
    if model_name == "PretrainedResNet152":
        return ResNet(152,
                      num_classes,
                      args.optimizer,
                      args.learning_rate,
                      l2_penalty=args.l2_penalty)

    raise Exception("Wrong model name provided!")
