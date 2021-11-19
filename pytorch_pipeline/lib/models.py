"""
Module to create the models architectures.
"""
import argparse

from torch import nn
import pytorch_lightning as pl
import torchvision.models as models

from .training import ImageClassifier


class ResNet(nn.Module):
    """
    Class to create a model using transfer learning from a ResNet model.
    """

    def __init__(self,
                 resnet_version: int,
                 num_classes: int,
                 pretrained: bool = True):
        """
        Model constructor.

        Args:
            resnet_version: Choices: [18, 34, 50, 101, 152]

            num_classes: Number of units to put in the last Dense layer.

            pretrained: To use or not the pretrained weights from imagenet
        """
        super(ResNet, self).__init__()

        self.pretrained = pretrained  # We need this for the pipeline

        if resnet_version == 18:
            backbone = models.resnet18(pretrained=pretrained)
        elif resnet_version == 34:
            backbone = models.resnet34(pretrained=pretrained)
        elif resnet_version == 50:
            backbone = models.resnet50(pretrained=pretrained)
        elif resnet_version == 101:
            backbone = models.resnet101(pretrained=pretrained)
        elif resnet_version == 152:
            backbone = models.resnet152(pretrained=pretrained)
        else:
            raise Exception(f"ResNet version '{resnet_version}' is not valid!")

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


class VGG(nn.Module):
    """
    Class to create a model using transfer learning from a VGG model.
    """

    def __init__(self,
                 vgg_version: str,
                 num_classes: int,
                 pretrained: bool = True):
        """
        Model constructor.

        Args:
            vgg_version: Choices: ["16", "16BN", "19", "19BN"]

            num_classes: Number of units to put in the last Dense layer.

            pretrained: To use or not the pretrained weights from imagenet
        """
        super(VGG, self).__init__()

        self.pretrained = pretrained  # We need this for the pipeline

        if vgg_version == "16":
            backbone = models.vgg16(pretrained=pretrained)
        elif vgg_version == "16BN":
            backbone = models.vgg16_bn(pretrained=pretrained)
        elif vgg_version == "19":
            backbone = models.vgg19(pretrained=pretrained)
        elif vgg_version == "19BN":
            backbone = models.vgg19_bn(pretrained=pretrained)
        else:
            raise Exception(f"VGG version '{vgg_version}' is not valid!")

        # Extract the pretrained convolutional part
        layers = list(backbone.children())[:-1]  # Get the convolutional part
        self.feature_extractor = nn.Sequential(*layers)

        # Prepare the new classifier
        in_features = list(backbone.classifier.children())[0].in_features
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        out = self.fully_connected(x)
        return out


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
        model = ResNet(18, num_classes, pretrained=False)
    elif model_name == "ResNet34":
        model = ResNet(34, num_classes, pretrained=False)
    elif model_name == "ResNet50":
        model = ResNet(50, num_classes, pretrained=False)
    elif model_name == "ResNet101":
        model = ResNet(101, num_classes, pretrained=False)
    elif model_name == "ResNet152":
        model = ResNet(152, num_classes, pretrained=False)
    # Pretrained ResNet models
    elif model_name == "PretrainedResNet18":
        model = ResNet(18, num_classes, pretrained=True)
    elif model_name == "PretrainedResNet34":
        model = ResNet(34, num_classes, pretrained=True)
    elif model_name == "PretrainedResNet50":
        model = ResNet(50, num_classes, pretrained=True)
    elif model_name == "PretrainedResNet101":
        model = ResNet(101, num_classes, pretrained=True)
    elif model_name == "PretrainedResNet152":
        model = ResNet(152, num_classes, pretrained=True)
    # NO pretrained VGG models
    elif model_name == "VGG16":
        model = VGG("16", num_classes, pretrained=False)
    elif model_name == "VGG16BN":
        model = VGG("16BN", num_classes, pretrained=False)
    elif model_name == "VGG19":
        model = VGG("19", num_classes, pretrained=False)
    elif model_name == "VGG19BN":
        model = VGG("19BN", num_classes, pretrained=False)
    # Pretrained VGG models
    elif model_name == "PretrainedVGG16":
        model = VGG("16", num_classes, pretrained=True)
    elif model_name == "PretrainedVGG16BN":
        model = VGG("16BN", num_classes, pretrained=True)
    elif model_name == "PretrainedVGG19":
        model = VGG("19", num_classes, pretrained=True)
    elif model_name == "PretrainedVGG19BN":
        model = VGG("19BN", num_classes, pretrained=True)
    else:
        raise Exception("Wrong model name provided!")

    return ImageClassifier(model,
                           args.optimizer,
                           args.learning_rate,
                           l2_penalty=args.l2_penalty)
