"""
Module to create the models architectures.
"""
import os
from typing import Callable
import urllib.request

import pyeddl.eddl as eddl


##########
# ResNet #
##########

def conv_bn_relu(in_layer,  # A EDDL layer
                 filters: int,
                 kernel_size: tuple = (3, 3),
                 strides: tuple = (1, 1),
                 padding: str = "same"):
    """
    Builds the sequence of layers: Conv2D -> BatchNormalization -> ReLu.

    Args:
        in_layer: Input layer of the Conv2D.

        **args: The parameters for the Conv2D layer.

    Returns:
        The reference to the last layer of the sequence (ReLu).
    """
    conv = eddl.Conv(in_layer, filters, kernel_size, strides, padding)
    bn = eddl.BatchNormalization(conv, affine=True)
    relu = eddl.ReLu(bn)

    return relu


def bn_relu_conv(in_layer,  # A EDDL layer
                 filters: int,
                 kernel_size: tuple = (3, 3),
                 strides: tuple = (1, 1),
                 padding: str = "same"):
    """
    Builds the sequence of layers: BatchNormalization -> ReLu -> Conv2D.

    Args:
        in_layer: Input layer of the Conv2D.

        **args: The parameters for the Conv2D layer.

    Returns:
        The reference to the last layer of the sequence (Conv2D).
    """
    bn = eddl.BatchNormalization(in_layer, affine=True)
    relu = eddl.ReLu(bn)
    conv = eddl.Conv(relu, filters, kernel_size, strides, padding)

    return conv


def shortcut(in_layer, residual):
    """
    Given the input of a residual block and its residual output, this function
    creates the shortcut connection (with an Add layer) and fixes the
    input tensor size (height, width) or channels if needed.

    Args:
        in_layer: An EDDL layer.

        residual: An EDDL layer.

    Returns:
        The reference to the Add layer that performs the shortcut.
    """
    # Check if the shapes match
    in_ch, in_h, in_w = in_layer.output.shape[-3:]
    res_ch, res_h, res_w = residual.output.shape[-3:]
    # The stride values should be integers after division
    stride_h = in_h // res_h
    stride_w = in_w // res_w
    eq_channels = in_ch == res_ch

    shortcut = in_layer
    if stride_h > 1 or stride_w > 1 or not eq_channels:
        # Use a 1x1 Conv to fix the shape
        shortcut = eddl.Conv(in_layer,
                             res_ch,
                             (1, 1),
                             (stride_h, stride_w),
                             "valid")

    return eddl.Add([shortcut, residual])


def basic_block(in_layer,  # A EDDL Layer
                filters: int,
                strides: tuple = (1, 1),
                is_first_layer: bool = False):
    """
    Builds a basic convolutional block with a shortcut at the end to add the
    residual connection.

    Args:
        in_layer: Input layer of the convolutional block.

        filters: Number of filters for the Conv layer.

        strides: Strides of the Conv layer. The shapes are then fixed in the
                 shortcut at the end of the block.

        is_first_layer: To avoid using preactivation for the first layer of
                        the first convolutional block. Because the previous
                        layers are BN -> ReLu -> MaxPool2D.

    Returns:
        A reference to the last layer of the block (after the shortcut).
    """
    l = in_layer  # Aux layer to save the input reference for the shortcut
    if strides == (2, 2):
        l = eddl.Pad(l, [0, 1, 1, 0])  # Fix asymmetric padding
        padding = "valid"
    else:
        padding = "same"

    if is_first_layer:
        # Avoid preactivation
        l = eddl.Conv(l, filters, (3, 3), strides, padding)
    else:
        l = bn_relu_conv(l, filters, (3, 3), strides, padding)

    residual = bn_relu_conv(l, filters, (3, 3), (1, 1), "same")

    return shortcut(in_layer, residual)


def bottleneck_block(in_layer,  # A EDDL Layer
                     filters: int,
                     strides: tuple = (1, 1),
                     is_first_layer: bool = False):
    """
    Builds a bottleneck convolutional block with a shortcut at the end
    to add the residual connection.

    Args:
        in_layer: Input layer of the convolutional block.

        filters: Number of kernels for the first 2 Conv layers. The third
                 layer uses "filters * 4" kernels.

        strides: Strides of the Conv layer. The shapes are then fixed in the
                 shortcut at the end of the block.

        is_first_layer: To avoid using preactivation for the first layer of
                        the first convolutional block. Because the previous
                        layers are BN -> ReLu -> MaxPool2D.

    Returns:
        A reference to the last layer of the block (after the shortcut).
    """
    if is_first_layer:
        # Avoid preactivation
        l = eddl.Conv(in_layer, filters, (1, 1), strides, "same")
    else:
        l = bn_relu_conv(in_layer, filters, (1, 1), strides, "same")

    l = bn_relu_conv(l, filters, (3, 3), (1, 1), "same")
    residual = bn_relu_conv(l, filters * 4, (1, 1), (1, 1), "same")

    return shortcut(in_layer, residual)


def residual_block(in_layer,  # A EDDL layer
                   conv_block: Callable,
                   filters: int,
                   n_blocks: int,
                   is_first_layer: bool = False):
    """
    Builds a residual block based on the convolutional block provided.

    Args:
        in_layer: Input layer for the block.

        conv_block: Basic convolutional block to repeat in order to build
                    the residual block.

        filters: Filters for the basic convolutional block.

        n_blocks: Number of repetitions of the basic convolutional block.

        is_first_layer: To avoid using preactivation for the first layer of
                        the first convolutional block. Because the previous
                        layers are BN -> ReLu -> MaxPool2D.

    Returns:
        The reference to the last layer of the residual block.
    """
    l = in_layer  # Auxiliary reference
    for b in range(n_blocks):
        strides = (1, 1)
        if b == 0 and not is_first_layer:
            # Add reduction in the fist layer of each residual block
            strides = (2, 2)

        l = conv_block(l, filters, strides, (is_first_layer and b == 0))

    return l


def build_resnet(in_shape: tuple,
                 num_classes: int,
                 block_type: Callable,
                 n_blocks: list,
                 multiclass: bool) -> list:
    """
    Parametrized constructor to build every variant of the ResNet architecture.

    Args:
        in_shape: Input shape of the model (without batch dimension).

        num_classes: Number of units in the last Dense layer.

        block_type: Function that creates a convolutional block.

        n_blocks: List of ints to determine the number of convolutional blocks
                  at each level of the model.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object with the ResNet architecture.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    in_ = eddl.Input(in_shape)
    # First conv block before the resiual blocks
    l = eddl.Pad(in_, [3, 3, 3, 3])  # Fix asymmetric padding
    l = conv_bn_relu(l, 64, (7, 7), (2, 2), "valid")
    l = eddl.Pad(l, [1, 1, 1, 1])  # Fix asymmetric padding
    l = eddl.MaxPool2D(l, (3, 3), (2, 2), "valid")

    # Build residual blocks
    filters = 64
    for block_idx, n_blocks in enumerate(n_blocks):
        l = residual_block(in_layer=l,
                           conv_block=block_type,
                           filters=filters,
                           n_blocks=n_blocks,
                           is_first_layer=(block_idx == 0))
        filters *= 2

    # Activation before densely connected part
    l = eddl.BatchNormalization(l, affine=True)
    l = eddl.ReLu(l)

    l = eddl.GlobalAveragePool2D(l)
    l = eddl.Flatten(l)
    l = eddl.Dense(l, num_classes)
    if multiclass:
        out_ = eddl.Sigmoid(l)
    else:
        out_ = eddl.Softmax(l)

    return eddl.Model([in_], [out_]), True, []


def resnet_18(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    return build_resnet(in_shape, num_classes, basic_block, [2, 2, 2, 2], multiclass)


def resnet_34(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    return build_resnet(in_shape, num_classes, basic_block, [3, 4, 6, 3], multiclass)


def resnet_50(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    return build_resnet(in_shape, num_classes, bottleneck_block, [3, 4, 6, 3], multiclass)


def resnet_101(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    return build_resnet(in_shape, num_classes, bottleneck_block, [3, 4, 23, 3], multiclass)


def resnet_152(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    return build_resnet(in_shape, num_classes, bottleneck_block, [3, 8, 36, 3], multiclass)


#################
# CUSTOM MODELS #
#################

def model_1(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    """
    Creates an EDDL model with the topology 'model_1'

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    in_ = eddl.Input(in_shape)

    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, 32, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 64, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 64, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 128, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 128, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 256, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.Flatten(l)
    l = eddl.ReLu(eddl.Dense(l, 512))
    l = eddl.Dense(l, num_classes)
    if multiclass:
        out_ = eddl.Sigmoid(l)
    else:
        out_ = eddl.Softmax(l)

    return eddl.Model([in_], [out_]), True, []


def model_2(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    """
    Creates an EDDL model with the topology 'model_2'

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    in_ = eddl.Input(in_shape)

    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, 32, [5, 5]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 64, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 64, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 128, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 128, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 256, [3, 3]), True))
    l = eddl.MaxPool2D(l, [2, 2])
    l = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l, 256, [3, 3]), True))
    l = eddl.GlobalAveragePool2D(l)
    l = eddl.Flatten(l)
    l = eddl.ReLu(eddl.Dense(l, 128))
    l = eddl.Dense(l, num_classes)
    if multiclass:
        out_ = eddl.Sigmoid(l)
    else:
        out_ = eddl.Softmax(l)

    return eddl.Model([in_], [out_]), True, []


def model_3(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    """
    Creates an EDDL model with the topology 'model_3'

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    in_ = eddl.Input(in_shape)

    block1 = conv_bn_relu(in_, filters=32, kernel_size=[5, 5])
    block1 = conv_bn_relu(block1, filters=32, kernel_size=[3, 3])
    block1 = eddl.Add([block1, eddl.PointwiseConv2D(in_, 32)])
    block1 = eddl.MaxPool2D(block1, [2, 2])

    block2 = conv_bn_relu(block1, filters=64, kernel_size=[3, 3])
    block2 = conv_bn_relu(block2, filters=64, kernel_size=[3, 3])
    block2 = eddl.Add([block2, eddl.PointwiseConv2D(block1, 64)])
    block2 = eddl.MaxPool2D(block2, [2, 2])

    block3 = conv_bn_relu(block2, filters=64, kernel_size=[3, 3])
    block3 = conv_bn_relu(block3, filters=64, kernel_size=[3, 3])
    block3 = eddl.Add([block3, block2])
    block3 = eddl.MaxPool2D(block3, [2, 2])

    block4 = conv_bn_relu(block3, filters=128, kernel_size=[3, 3])
    block4 = conv_bn_relu(block4, filters=128, kernel_size=[3, 3])
    block4 = eddl.Add([block4, eddl.PointwiseConv2D(block3, 128)])
    block4 = eddl.MaxPool2D(block4, [2, 2])

    block5 = conv_bn_relu(block4, filters=128, kernel_size=[3, 3])
    block5 = conv_bn_relu(block5, filters=128, kernel_size=[3, 3])
    block5 = eddl.Add([block5, block4])
    block5 = eddl.MaxPool2D(block5, [2, 2])

    block6 = conv_bn_relu(block5, filters=256, kernel_size=[3, 3])
    block6 = conv_bn_relu(block6, filters=256, kernel_size=[3, 3])
    block6 = eddl.Add([block6, eddl.PointwiseConv2D(block5, 256)])
    block6 = eddl.MaxPool2D(block6, [2, 2])

    block7 = conv_bn_relu(block6, filters=256, kernel_size=[3, 3])
    block7 = conv_bn_relu(block7, filters=256, kernel_size=[3, 3])
    block7 = eddl.Add([block7, block6])
    block7 = eddl.GlobalAveragePool2D(block7)
    conv_out = eddl.Flatten(block7)

    dense1 = eddl.ReLu(eddl.Dense(conv_out, 128))
    dense2 = eddl.Dense(dense1, num_classes)
    if multiclass:
        out_ = eddl.Sigmoid(dense2)
    else:
        out_ = eddl.Softmax(dense2)

    return eddl.Model([in_], [out_]), True, []


def model_4(in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    """
    Creates an EDDL model with the topology 'model_4'

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    in_ = eddl.Input(in_shape)

    block0 = eddl.Pad(in_, [2, 3, 3, 2])
    block0 = conv_bn_relu(
        block0, filters=32, kernel_size=[7, 7], strides=(2, 2))
    block0 = conv_bn_relu(
        block0, filters=64, kernel_size=[5, 5], strides=(2, 2))

    block1 = conv_bn_relu(block0, filters=64, kernel_size=[3, 3])
    block1 = conv_bn_relu(block1, filters=64, kernel_size=[3, 3])
    block1 = eddl.Add([block1, block0])
    block1 = eddl.MaxPool2D(block1, [2, 2])

    block2 = conv_bn_relu(block1, filters=96, kernel_size=[3, 3])
    block2 = conv_bn_relu(block2, filters=96, kernel_size=[3, 3])
    block2 = eddl.Add([block2, eddl.PointwiseConv2D(block1, 96)])
    block2 = eddl.MaxPool2D(block2, [2, 2])

    block3 = conv_bn_relu(block2, filters=128, kernel_size=[3, 3])
    block3 = conv_bn_relu(block3, filters=128, kernel_size=[3, 3])
    block3 = eddl.Add([block3, eddl.PointwiseConv2D(block2, 128)])
    block3 = eddl.MaxPool2D(block3, [2, 2])

    block4 = conv_bn_relu(block3, filters=160, kernel_size=[3, 3])
    block4 = conv_bn_relu(block4, filters=160, kernel_size=[3, 3])
    block4 = eddl.Add([block4, eddl.PointwiseConv2D(block3, 160)])
    block4 = eddl.MaxPool2D(block4, [2, 2])

    block5 = conv_bn_relu(block4, filters=192, kernel_size=[3, 3])
    block5 = conv_bn_relu(block5, filters=192, kernel_size=[3, 3])
    block5 = eddl.Add([block5, eddl.PointwiseConv2D(block4, 192)])
    block5 = eddl.MaxPool2D(block5, [2, 2])

    block6 = conv_bn_relu(block5, filters=224, kernel_size=[3, 3])
    block6 = conv_bn_relu(block6, filters=224, kernel_size=[3, 3])
    block6 = eddl.Add([block6, eddl.PointwiseConv2D(block5, 224)])
    block6 = eddl.MaxPool2D(block6, [2, 2])

    block7 = conv_bn_relu(block6, filters=256, kernel_size=[3, 3])
    block7 = conv_bn_relu(block7, filters=256, kernel_size=[3, 3])
    block7 = eddl.Add([block7, eddl.PointwiseConv2D(block6, 256)])

    block8 = conv_bn_relu(
        block7, filters=512, kernel_size=[3, 3], padding="valid")
    block8 = conv_bn_relu(block8, filters=512, kernel_size=[1, 1])
    conv_out = eddl.Flatten(block8)

    drop = eddl.Dropout(conv_out, 0.4)
    dense1 = eddl.ReLu(eddl.Dense(drop, 1024))
    dense2 = eddl.Dense(dense1, num_classes)
    if multiclass:
        out_ = eddl.Sigmoid(dense2)
    else:
        out_ = eddl.Softmax(dense2)

    return eddl.Model([in_], [out_]), True, []


def pretrained_resnet(in_shape: tuple,
                      num_classes: int,
                      version: str,
                      multiclass: bool) -> list:
    """
    Uses a pretrained ResNet to extract the convolutional block and then
    append a new densely connected part to do the classification.

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        version: A string to select the version of the ResNet model to use.
                 Versions available: "18", "34", "50", "101" and "152"

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    # Import the pretrained ResNet model without the densely connected part
    #   Note: the last layer name is "top"
    if version == "18":
        pretrained_model = eddl.download_resnet18(input_shape=in_shape)
    elif version == "34":
        pretrained_model = eddl.download_resnet34(input_shape=in_shape)
    elif version == "50":
        pretrained_model = eddl.download_resnet50(input_shape=in_shape)
    elif version == "101":
        pretrained_model = eddl.download_resnet101(input_shape=in_shape)
    elif version == "152":
        pretrained_model = eddl.download_resnet152(input_shape=in_shape)

    # Get the reference to the input layer of the pretrained model
    in_ = eddl.getLayer(pretrained_model, "input")
    # Get the reference to the last layer of the pretrained model
    l = eddl.getLayer(pretrained_model, "top")

    # Create the new densely connected part
    input_units = l.output.shape[-1]
    l = eddl.Dense(l, input_units // 2, name="dense1")
    l = eddl.ReLu(l, name="dense1_relu")
    l = eddl.Dropout(l, 0.4, name="dense1_dropout")
    l = eddl.Dense(l, num_classes, name="dense_out")
    if multiclass:
        out_ = eddl.Sigmoid(l)
    else:
        out_ = eddl.Softmax(l)

    return eddl.Model([in_], [out_]), False, ["dense1", "dense_out"]


def pretrained_vgg16(in_shape: tuple,
                     num_classes: int,
                     multiclass: bool) -> list:
    """
    Uses a pretrained VGG16 to extract the convolutional block and then
    append a new densely connected part to do the classification.

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    # Import the pretrained VGG16 model without the densely connected part
    #   Note: the last layer name is "top"
    pretrained_model = eddl.download_vgg16(input_shape=in_shape)

    # Get the reference to the input layer of the pretrained model
    in_ = eddl.getLayer(pretrained_model, "input")
    # Get the reference to the last layer of the pretrained conv block
    l = eddl.getLayer(pretrained_model, "top")

    # Create the new densely connected part
    input_units = l.output.shape[-1]
    # Dense 1
    l = eddl.Dense(l, 4096, name="dense1")
    l = eddl.ReLu(l, name="dense1_relu")
    l = eddl.Dropout(l, 0.4, name="dense1_dropout")
    # Dense 2
    l = eddl.Dense(l, 4096, name="dense2")
    l = eddl.ReLu(l, name="dense2_relu")
    l = eddl.Dropout(l, 0.4, name="dense2_dropout")
    # Dense 3
    l = eddl.Dense(l, num_classes, name="dense_out")
    if multiclass:
        out_ = eddl.Sigmoid(l)
    else:
        out_ = eddl.Softmax(l)

    # This layers must be initialized because they are not pretrained
    layer2init = ["dense1", "dense2", "dense_out"]

    return eddl.Model([in_], [out_]), False, layer2init


def pretrained_vgg16BN(in_shape: tuple,
                       num_classes: int,
                       multiclass: bool) -> list:
    """
    Uses a pretrained VGG16 with BN to extract the convolutional block and then
    append a new densely connected part to do the classification.

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    # Import the pretrained VGG16-BN model without the densely connected part
    #   Note: the last layer name is "top"
    pretrained_model = eddl.download_vgg16_bn(input_shape=in_shape)

    # Get the reference to the input layer of the pretrained model
    in_ = eddl.getLayer(pretrained_model, "input")
    # Get the reference to the last layer of the pretrained conv block
    l = eddl.getLayer(pretrained_model, "top")

    # Create the new densely connected part
    input_units = l.output.shape[-1]
    # Dense 1
    l = eddl.Dense(l, 4096, name="dense1")
    l = eddl.ReLu(l, name="dense1_relu")
    l = eddl.Dropout(l, 0.4, name="dense1_dropout")
    # Dense 2
    l = eddl.Dense(l, 4096, name="dense2")
    l = eddl.ReLu(l, name="dense2_relu")
    l = eddl.Dropout(l, 0.4, name="dense2_dropout")
    # Dense 3
    l = eddl.Dense(l, num_classes, name="dense_out")
    if multiclass:
        out_ = eddl.Sigmoid(l)
    else:
        out_ = eddl.Softmax(l)

    # This layers must be initialized because they are not pretrained
    layer2init = ["dense1", "dense2", "dense_out"]

    return eddl.Model([in_], [out_]), False, layer2init


def pretrained_vgg19(in_shape: tuple,
                     num_classes: int,
                     multiclass: bool) -> list:
    """
    Uses a pretrained VGG19 to extract the convolutional block and then
    append a new densely connected part to do the classification.

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    # Import the pretrained VGG19 model without the densely connected part
    #   Note: the last layer name is "top"
    pretrained_model = eddl.download_vgg19(input_shape=in_shape)

    # Get the reference to the input layer of the pretrained model
    in_ = eddl.getLayer(pretrained_model, "input")
    # Get the reference to the last layer of the pretrained conv block
    l = eddl.getLayer(pretrained_model, "top")

    # Create the new densely connected part
    input_units = l.output.shape[-1]
    # Dense 1
    l = eddl.Dense(l, 4096, name="dense1")
    l = eddl.ReLu(l, name="dense1_relu")
    l = eddl.Dropout(l, 0.4, name="dense1_dropout")
    # Dense 2
    l = eddl.Dense(l, 4096, name="dense2")
    l = eddl.ReLu(l, name="dense2_relu")
    l = eddl.Dropout(l, 0.4, name="dense2_dropout")
    # Dense 3
    l = eddl.Dense(l, num_classes, name="dense_out")
    if multiclass:
        out_ = eddl.Sigmoid(l)
    else:
        out_ = eddl.Softmax(l)

    # This layers must be initialized because they are not pretrained
    layer2init = ["dense1", "dense2", "dense_out"]

    return eddl.Model([in_], [out_]), False, layer2init


def pretrained_vgg19BN(in_shape: tuple,
                       num_classes: int,
                       multiclass: bool) -> list:
    """
    Uses a pretrained VGG19 with BN to extract the convolutional block and then
    append a new densely connected part to do the classification.

    Args:
        in_shape: Input shape of the model (channels, height, width).

        num_classes: Number of units for the output Dense layer.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object.

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    # Import the pretrained VGG19-BN model without the densely connected part
    #   Note: the last layer name is "top"
    pretrained_model = eddl.download_vgg19_bn(input_shape=in_shape)

    # Get the reference to the input layer of the pretrained model
    in_ = eddl.getLayer(pretrained_model, "input")
    # Get the reference to the last layer of the pretrained conv block
    l = eddl.getLayer(pretrained_model, "top")

    # Create the new densely connected part
    input_units = l.output.shape[-1]
    # Dense 1
    l = eddl.Dense(l, 4096, name="dense1")
    l = eddl.ReLu(l, name="dense1_relu")
    l = eddl.Dropout(l, 0.4, name="dense1_dropout")
    # Dense 2
    l = eddl.Dense(l, 4096, name="dense2")
    l = eddl.ReLu(l, name="dense2_relu")
    l = eddl.Dropout(l, 0.4, name="dense2_dropout")
    # Dense 3
    l = eddl.Dense(l, num_classes, name="dense_out")
    if multiclass:
        out_ = eddl.Sigmoid(l)
    else:
        out_ = eddl.Softmax(l)

    # This layers must be initialized because they are not pretrained
    layer2init = ["dense1", "dense2", "dense_out"]

    return eddl.Model([in_], [out_]), False, layer2init


def get_model(model_name: str, in_shape: tuple, num_classes: int, multiclass: bool) -> list:
    """
    Auxiliary function to create the selected model topology.

    Args:
        model_name: Name of the architecture of the model to create.

        in_shape: Tuple with the input shape of the model (without batch dim).

        num_classes: Number of units in the last layer for classification.

        multiclass: If True uses sigmoid in the output layer, else uses Softmax.

    Returns:
        A list with:
            - The EDDL model object (not built).

            - Boolean to indicate if the weights must be initialized.

            - A list with the layer names that should be initialized in case
              of passing False in the previous output value. Useful when using
              a pretrained convolutional block followed by a new set of Dense
              layers for classification.
    """
    # Custom models
    if model_name == "model_1":
        return model_1(in_shape, num_classes, multiclass)
    if model_name == "model_2":
        return model_2(in_shape, num_classes, multiclass)
    if model_name == "model_3":
        return model_3(in_shape, num_classes, multiclass)
    if model_name == "model_4":
        return model_4(in_shape, num_classes, multiclass)

    # ResNet models (from scratch)
    if model_name == "ResNet18":
        return resnet_18(in_shape, num_classes, multiclass)
    if model_name == "ResNet34":
        return resnet_34(in_shape, num_classes, multiclass)
    if model_name == "ResNet50":
        return resnet_50(in_shape, num_classes, multiclass)
    if model_name == "ResNet101":
        return resnet_101(in_shape, num_classes, multiclass)
    if model_name == "ResNet152":
        return resnet_152(in_shape, num_classes, multiclass)

    # ResNet models (pretrained from ONNX)
    if model_name == "Pretrained_ResNet18":
        return pretrained_resnet(in_shape, num_classes, "18", multiclass)
    if model_name == "Pretrained_ResNet34":
        return pretrained_resnet(in_shape, num_classes, "34", multiclass)
    if model_name == "Pretrained_ResNet50":
        return pretrained_resnet(in_shape, num_classes, "50", multiclass)
    if model_name == "Pretrained_ResNet101":
        return pretrained_resnet(in_shape, num_classes, "101", multiclass)
    if model_name == "Pretrained_ResNet152":
        return pretrained_resnet(in_shape, num_classes, "152", multiclass)

    # VGG models
    if model_name == "Pretrained_VGG16":
        return pretrained_vgg16(in_shape, num_classes, multiclass)
    if model_name == "Pretrained_VGG16BN":
        return pretrained_vgg16BN(in_shape, num_classes, multiclass)
    if model_name == "Pretrained_VGG19":
        return pretrained_vgg19(in_shape, num_classes, multiclass)
    if model_name == "Pretrained_VGG19BN":
        return pretrained_vgg19BN(in_shape, num_classes, multiclass)

    raise Exception("Wrong model name provided!")


#####################
# Transfer learning #
#####################

def extract_pretrained_layers(model: eddl.Model,
                              layers2remove: list,
                              input_name: str,
                              last_name: str) -> list:
    """
    This function removes the selected layers of a model to extract the desired
    pretrained block.
    Note: The input model will be modified.

    Args:
        model: Model to extract the layers from.

        layers2remove: List with the names of the layers that we are going
                       to remove.

        input_name: Name of the input layer of the model.

        last_name: Name of the last layer of the pretrained block that we want
                   to extract.

    Returns:
        A list with:
            - A reference to the input layer of the model.

            - A reference to the last layer of the pretrained block.
    """
    # Remove the selected layers
    for layer_name in layers2remove:
        eddl.removeLayer(model, layer_name)

    # Get the reference to the input layer of the model
    in_ = eddl.getLayer(model, input_name)
    # Get the reference to the last layer of the convolutional part
    out_ = eddl.getLayer(model, last_name)

    return in_, out_


def get_model_tl(onnx_file: str,
                 model_name: str,
                 in_shape: tuple,
                 num_classes: int) -> list:
    """
    Auxiliary function to use transfer learning to load the ONNX provided and
    change the fully connected layers for the new classification task.

    Args:
        onnx_file: Path to the ONNX file to load.

        model_name: Name of the architecture used to create the ONNX file.

        in_shape: Tuple with the input shape of the model (without batch dim).

        num_classes: Number of units in the last layer for classification.

    Returns:
        A list with:
            - The EDDL model object (not built).

            - A list with the layer names that should be initialized (usually
              the new dense layers added).
    """
    # Load the model from ONNX
    pretrained_model = eddl.import_net_from_onnx_file(onnx_file,
                                                      input_shape=in_shape)
    if model_name == "ResNet50":
        layers2remove = ["softmax50", "dense1"]
        in_, conv_out = extract_pretrained_layers(pretrained_model,
                                                  layers2remove,
                                                  "input1",
                                                  "reshape1")
    else:
        raise Exception("Wrong model name provided!")

    # Add the new dense layer for classification
    input_units = conv_out.output.shape[-1]  # conv_out is a Flatten layer
    l = eddl.Dense(conv_out, input_units // 4, name="dense1")
    l = eddl.BatchNormalization(l, True, name="dense1_bn")
    l = eddl.ReLu(l, name="dense1_relu")
    out_ = eddl.Softmax(eddl.Dense(l, num_classes, name="dense_out"))

    # This layers must be initialized because they are not pretrained
    layer2init = ["dense1", "dense1_bn", "dense_out"]

    return eddl.Model([in_], [out_]), layer2init
