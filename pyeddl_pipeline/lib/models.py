"""
Module to create the models architectures.
"""
from typing import Callable

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
                 n_blocks: list):
    """
    Parametrized constructor to build every variant of the ResNet architecture.

    Args:
        in_shape: Input shape of the model (without batch dimension).

        num_classes: Number of units in the last Dense layer.

        block_type: Function that creates a convolutional block.

        n_blocks: List of ints to determine the number of convolutional blocks
                  at each level of the model.

    Returns:
        A EDDL model with the defined ResNet architecture.
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
    out_ = eddl.Softmax(l)

    return eddl.Model([in_], [out_])


def resnet_18(in_shape: tuple, num_classes: int) -> eddl.Model:
    return build_resnet(in_shape, num_classes, basic_block, [2, 2, 2, 2])


def resnet_34(in_shape: tuple, num_classes: int) -> eddl.Model:
    return build_resnet(in_shape, num_classes, basic_block, [3, 4, 6, 3])


def resnet_50(in_shape: tuple, num_classes: int) -> eddl.Model:
    return build_resnet(in_shape, num_classes, bottleneck_block, [3, 4, 6, 3])


def resnet_101(in_shape: tuple, num_classes: int) -> eddl.Model:
    return build_resnet(in_shape, num_classes, bottleneck_block, [3, 4, 23, 3])


def resnet_152(in_shape: tuple, num_classes: int) -> eddl.Model:
    return build_resnet(in_shape, num_classes, bottleneck_block, [3, 8, 36, 3])


#################
# CUSTOM MODELS #
#################

def model_1(in_shape: tuple, num_classes: int) -> eddl.Model:
    """Creates an EDDL model with the topology 'model_1'"""
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
    out_ = eddl.Softmax(eddl.Dense(l, num_classes))

    return eddl.Model([in_], [out_])


def model_2(in_shape: tuple, num_classes: int) -> eddl.Model:
    """Creates an EDDL model with the topology 'model_2'"""
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
    out_ = eddl.Softmax(eddl.Dense(l, num_classes))

    return eddl.Model([in_], [out_])


def model_3(in_shape: tuple, num_classes: int) -> eddl.Model:
    """Creates an EDDL model with the topology 'model_3'"""
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
    out_ = eddl.Softmax(eddl.Dense(dense1, num_classes))

    return eddl.Model([in_], [out_])


def get_model(model_name: str, in_shape: tuple, num_classes: int) -> eddl.Model:
    """
    Auxiliary function to create the selected model topology.

    Args:
        model_name: Name of the architecture of the model to create.

        in_shape: Tuple with the input shape of the model (without batch dim).

        num_classes: Number of units in the last layer for classification.

    Returns:
        An EDDL model (not built).
    """
    # Custom models
    if model_name == "model_1":
        return model_1(in_shape, num_classes)
    if model_name == "model_2":
        return model_2(in_shape, num_classes)
    if model_name == "model_3":
        return model_3(in_shape, num_classes)

    # ResNet models
    if model_name == "ResNet18":
        return resnet_18(in_shape, num_classes)
    if model_name == "ResNet34":
        return resnet_34(in_shape, num_classes)
    if model_name == "ResNet50":
        return resnet_50(in_shape, num_classes)
    if model_name == "ResNet101":
        return resnet_101(in_shape, num_classes)
    if model_name == "ResNet152":
        return resnet_152(in_shape, num_classes)

    raise Exception("Wrong model name provided!")
