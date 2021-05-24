"""
Module to create the models architectures.
"""
import pyeddl.eddl as eddl
from pyeddl.eddl import Conv, BatchNormalization, ReLu, MaxPool2D
from pyeddl.eddl import Flatten, Dense, Softmax


def model_1(in_shape: tuple, num_classes: int) -> eddl.Model:
    """Creates an EDDL model with the topology 'model_1'"""
    in_ = eddl.Input(in_shape)

    l = ReLu(BatchNormalization(Conv(in_, 32, [3, 3]), True))
    l = MaxPool2D(l, [2, 2])
    l = ReLu(BatchNormalization(Conv(l, 64, [3, 3]), True))
    l = MaxPool2D(l, [2, 2])
    l = ReLu(BatchNormalization(Conv(l, 64, [3, 3]), True))
    l = MaxPool2D(l, [2, 2])
    l = ReLu(BatchNormalization(Conv(l, 128, [3, 3]), True))
    l = MaxPool2D(l, [2, 2])
    l = ReLu(BatchNormalization(Conv(l, 128, [3, 3]), True))
    l = MaxPool2D(l, [2, 2])
    l = ReLu(BatchNormalization(Conv(l, 256, [3, 3]), True))
    l = MaxPool2D(l, [2, 2])
    l = Flatten(l)
    l = ReLu(Dense(l, 512))
    out_ = Softmax(Dense(l, num_classes))

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
    if model_name == "model_1":
        return model_1(in_shape, num_classes)

    raise Exception("Wrong model name provided!")
