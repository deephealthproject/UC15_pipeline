"""
Module to create the models architectures.
"""
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from torchmetrics import Accuracy


##########
# ResNet #
##########

class Flatten(nn.Module):
    """
    Wrapper module to perform a flatten operation.
    """

    def forward(self, x):
        batch = x.size(0)
        return x.view((batch, -1))


class ConvBNReLU(nn.Module):
    """
    Module to create the sequence of layers Conv2D -> BatchNorm -> ReLU
    """

    def __init__(self,
                 in_channels: int,
                 filters: int,
                 kernel_size: tuple = (3, 3),
                 stride: tuple = 1,
                 padding: int = 1):
        """
        Constructor.

        Args:
            in_channels: Number of channels in the input tensor.

            **args: The params for the Conv2D layer.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,
                      filters,
                      kernel_size,
                      stride,
                      padding),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class BNReLUConv(nn.Module):
    """
    Module to create the sequence of layers BatchNorm -> ReLU -> Conv2D
    """

    def __init__(self,
                 in_channels: int,
                 filters: int,
                 kernel_size: tuple = (3, 3),
                 stride: tuple = 1,
                 padding: tuple = 1):
        """
        Constructor.

        Args:
            in_channels: Number of channels in the input tensor.

            **args: The params for the Conv2D layer.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=filters,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding)
        )

    def forward(self, x):
        return self.layers(x)


class Shortcut(nn.Module):
    """
    Module to build the shortcut connection at the end of each block.
    This module can add a Conv2d layer to fix the input shape of the
    input tensor to match the shape of the residual tensor.
    """

    def __init__(self,
                 add_conv_shortcut: bool = False,
                 in_channels: int = None,
                 shortcut_channels: int = None,
                 shortcut_stride: int = None):
        """
        Constructor.

        Args:
            in_channels: The number of channels of the input tensor that is not
                         the resiual.

            add_conv_shortcut: To put a Conv2d layer to add a shortcut
                               connection that fixes the shapes of the input
                               tensors to be equal.

            shortcut_channels: Number of kernels in the Conv2d shortcut.
                               (Must be specified if add_conv_shortcut == True)

            shortcut_stride: Stride in the Conv2d shortcut.
                             (Must be specified if add_conv_shortcut == True)
        """
        super().__init__()

        self.add_conv_shortcut = add_conv_shortcut
        if self.add_conv_shortcut:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=shortcut_channels,
                                  kernel_size=(1, 1),
                                  stride=shortcut_stride,
                                  padding=0)

    def forward(self, x_in, x_residual):
        shortcut = x_in
        if self.add_conv_shortcut:
            shortcut = self.conv(x_in)

        return shortcut + x_residual


class BasicBlock(nn.Module):
    """
    Builds a basic convolutional block with a shortcut connection.
    """

    def __init__(self,
                 in_channels: int,
                 filters: int,
                 strides: tuple = (1, 1),
                 is_first_layer: bool = False):
        """
        Constructor.

        Args:
            in_channels: Number of channels in the input tensor.

            filters: Number of filters for the convolutional layers.

            strides: Strides for the convolutional layers.

            is_first_layer: To avoid using preactivation for the first layer of
                            the first convolutional block. Because the previous
                            layers are BN -> ReLu -> MaxPool2D.
        """
        super().__init__()

        if is_first_layer:
            self.conv1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=filters,
                                   kernel_size=(3, 3),
                                   stride=strides,
                                   padding=1)
        else:
            self.conv1 = BNReLUConv(in_channels,
                                    filters,
                                    kernel_size=(3, 3),
                                    stride=strides,
                                    padding=1)

        self.residual = BNReLUConv(filters,
                                   filters,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=1)

        if strides != (1, 1):
            # We are doing downsample
            self.shortcut = Shortcut(add_conv_shortcut=True,
                                     in_channels=in_channels,
                                     shortcut_channels=filters,
                                     shortcut_stride=strides)
        else:
            self.shortcut = Shortcut()  # Without downsample

    def forward(self, x):
        aux = self.conv1(x)
        res = self.residual(aux)
        return self.shortcut(x, res)


class BottleneckBlock(nn.Module):
    """
    Builds a bottleneck convolutional block with a shortcut connection.
    """

    def __init__(self,
                 in_channels: int,
                 filters: int,
                 strides: tuple = (1, 1),
                 is_first_layer: bool = False):
        """
        Constructor.

        Args:
            in_channels: Number of channels in the input tensor.

            filters: Number of filters for the convolutional layers.

            strides: Strides for the convolutional layers.

            is_first_layer: To avoid using preactivation for the first layer of
                            the first convolutional block. Because the previous
                            layers are BN -> ReLu -> MaxPool2D.
        """
        super().__init__()

        if is_first_layer:
            self.conv1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=filters,
                                   kernel_size=(1, 1),
                                   stride=strides,
                                   padding=0)
        else:
            self.conv1 = BNReLUConv(in_channels,
                                    filters,
                                    kernel_size=(1, 1),
                                    stride=strides,
                                    padding=0)

        self.conv2 = BNReLUConv(filters,
                                filters,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1)

        self.residual = BNReLUConv(filters,
                                   filters * 4,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0)

        self.shortcut = Shortcut(add_conv_shortcut=True,
                                 in_channels=in_channels,
                                 shortcut_channels=filters * 4,
                                 shortcut_stride=strides)

    def forward(self, x):
        aux = self.conv1(x)
        aux = self.conv2(aux)
        res = self.residual(aux)
        return self.shortcut(x, res)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 bottleneck_block: bool,
                 filters: int,
                 n_blocks: int,
                 is_first_layer: bool = False):
        """
        Builds a residual block based on the convolutional block provided.

        Args:
            in_channels: Number of channels in the input tensor.

            bottleneck_block: To use a BottleneckBlock or a BasicBlock.

            filters: Filters for the basic convolutional block.

            n_blocks: Number of repetitions of the basic convolutional block.

            is_first_layer: To avoid using preactivation for the first layer of
                            the first convolutional block. Because the previous
                            layers are BN -> ReLu -> MaxPool2D.
        """
        super().__init__()

        # Block to use in the residual part
        if bottleneck_block:
            block_type = BottleneckBlock
        else:
            block_type = BasicBlock

        layers = []
        aux_in_ch = in_channels
        for b in range(n_blocks):
            strides = (1, 1)
            if b == 0 and not is_first_layer:
                # Add reduction in the fist layer of each residual block
                strides = (2, 2)

            layers.append(block_type(aux_in_ch,
                                     filters,
                                     strides,
                                     (is_first_layer and b == 0)))

            # Change the input channels for the next block
            if bottleneck_block:
                aux_in_ch = 4 * filters
            else:
                aux_in_ch = filters

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResNet(pl.LightningModule):
    """
    Class to handle the creation of all the variants of the ResNet architecture.
    """

    def __init__(self,
                 in_shape: int,
                 num_classes: int,
                 bottleneck_block: bool,
                 n_blocks: list,
                 optimizer: str = "Adam",
                 learning_rate: float = 0.0001):
        """
        Model constructor.

        Args:
            in_shape: Input shape of the model (without the batch dimension).

            num_classes: Number of units to put in the last Dense layer.

            bottleneck_block: To use a BottleneckBlock or a BasicBlock.

            n_blocks: List of ints to determine the number of convolutional
                      blocks at each level of the model.

            optimizer: Optimizer type to use (choices: ["Adam", "SGD"]).

            learning_rate: Learning rate to use in the optimizer.
        """
        super().__init__()

        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.metric = Accuracy()

        self.entry_block = nn.Sequential(
            ConvBNReLU(in_shape[0],
                       64,
                       kernel_size=(7, 7),
                       stride=(2, 2),
                       padding=3),
            nn.MaxPool2d((3, 3), (2, 2), padding=1)
        )

        # Build the residual layers
        layers = []
        in_channels = 64
        filters = 64
        for block_idx, blocks in enumerate(n_blocks):
            layers.append(ResidualBlock(in_channels,
                                        bottleneck_block,
                                        filters,
                                        blocks,
                                        is_first_layer=(block_idx == 0)))
            if bottleneck_block:
                in_channels = filters * 4
            else:
                in_channels = filters

            filters *= 2

        self.res_block = nn.Sequential(*layers)

        self.last_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # GlobalAveragePool2D
            Flatten()
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(in_channels, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.entry_block(x)
        x = self.res_block(x)
        x = self.last_block(x)
        out = self.fully_connected(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)  # From one hot to indexes

        logits = self(x)  # Forward

        # Compute loss and metrics
        loss = F.cross_entropy(logits, y)
        acc = self.metric(logits, y)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)

        return loss  # Pytorch Lightning handles the backward

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)  # From one hot to indexes

        logits = self(x)  # Forward

        # Compute loss and metrics
        loss = F.cross_entropy(logits, y)
        acc = self.metric(logits, y)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)  # From one hot to indexes

        logits = self(x)  # Forward

        # Compute loss and metrics
        loss = F.cross_entropy(logits, y)
        acc = self.metric(logits, y)

        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            return Adam(self.parameters(), lr=self.learning_rate)
        if self.optimizer == "SGD":
            return SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

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
    # ResNet models
    if model_name == "ResNet18":
        return ResNet(in_shape,
                      num_classes,
                      bottleneck_block=False,
                      n_blocks=[2, 2, 2, 2],
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate)
    if model_name == "ResNet34":
        return ResNet(in_shape,
                      num_classes,
                      bottleneck_block=False,
                      n_blocks=[3, 4, 6, 3],
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate)
    if model_name == "ResNet50":
        return ResNet(in_shape,
                      num_classes,
                      bottleneck_block=True,
                      n_blocks=[3, 4, 6, 3],
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate)
    if model_name == "ResNet101":
        return ResNet(in_shape,
                      num_classes,
                      bottleneck_block=True,
                      n_blocks=[3, 4, 23, 3],
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate)
    if model_name == "ResNet152":
        return ResNet(in_shape,
                      num_classes,
                      bottleneck_block=True,
                      n_blocks=[3, 8, 36, 3],
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate)

    raise Exception("Wrong model name provided!")
