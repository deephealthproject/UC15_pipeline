"""
Module with utility functions for the training pipeline.
"""
import os
import argparse
import random

from tqdm import tqdm

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

#####################
# Data Augmentation #
#####################


def augmentations_v1_0(size: tuple) -> ecvl.DatasetAugmentations:
    """Returns the v1.0 augmentations for each split (train, val, test)"""
    height, width = size

    tr_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim((height, width), ecvl.InterpolationType.cubic)
    ])

    val_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim((height, width), ecvl.InterpolationType.cubic)
    ])

    te_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim((height, width), ecvl.InterpolationType.cubic)
    ])

    return ecvl.DatasetAugmentations([tr_augs, val_augs, te_augs])


def get_augmentations(version: str, size: tuple) -> ecvl.DatasetAugmentations:
    """
    Auxiliary function to get the selected set of data augmentations for a
    ECVL dataset.

    Args:
        version: A string with the version tag of the augmentations to select.

        size: A tuple (height, width) with the size to resize the images.

    Returns:
        A ecvl.DatasetAugmentations object that defines the augmentations for
        each split of the dataset.
    """
    if version == "v1.0":
        return augmentations_v1_0(size)

    raise Exception("Wrong augmentations version provided!")


##############
# Optimizers #
##############


def get_optimizer(opt_name: str, learning_rate: float):
    """
    Creates the EDDL optimizer selected by the name provided.

    Args:
        opt_name: Type of optimizer to select. Options: "Adam", "SGD".

        learning_rate: Learning rate of the optimizer.

    Returns:
        An EDDL optimizer.
    """
    if opt_name == "Adam":
        return eddl.adam(learning_rate)
    if opt_name == "SGD":
        return eddl.sgd(learning_rate, momentum=0.9)

    raise Exception("Wrong optimizer name provided!")


###########################
# Training loop functions #
###########################

def shuffle_training_split(dataset: ecvl.DLDataset):
    """
    Shuffles the training split of the dataset provided.

    Args:
        dataset: Dataset object to shuffle.

    Returns:
        None, the shuffling is inplace.
    """
    split_data = dataset.GetSplit(ecvl.SplitType.training)
    random.shuffle(split_data)
    dataset.split_.training_ = split_data


def train(model: eddl.Model,
          dataset: ecvl.DLDataset,
          args: argparse.Namespace) -> list:
    """
    Executes the main training loop. Performs training, validation and model
    checkpoints.

    Args:
        model: EDDL model to train. Must be already built.

        dataset: ECVL dataset to load the data for training.

        args: The argparse object with all the configuration data like:
              batch_size, epochs...

    Returns:
        A list with the history of the losses and metrics during all the
        training epochs.
    """
    n_train_samples = len(dataset.GetSplit(ecvl.SplitType.training))
    n_train_batches = n_train_samples // args.batch_size
    n_val_samples = len(dataset.GetSplit(ecvl.SplitType.validation))
    n_val_batches = n_val_samples // args.batch_size

    # Create auxiliary tensors to load the data
    x = Tensor([args.batch_size, *args.in_shape])  # Images
    y = Tensor([args.batch_size, args.num_classes])  # Labels

    best_acc = 0.0  # To track the best model

    # Experiment name
    exp_name = (f"{args.model}_DA-{args.augmentations}_opt-{args.optimizer}"
                f"_lr-{args.learning_rate}")
    # Check that the ckpts folder exists
    os.makedirs(args.models_ckpts, exist_ok=True)

    random.seed(args.seed)  # Seed for shuffling the data

    print(f"Going to traing for {args.epochs} epochs:")
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}:")
        ##################
        # Training phase #
        ##################

        # Prepare dataset
        dataset.SetSplit(ecvl.SplitType.training)
        shuffle_training_split(dataset)
        dataset.ResetAllBatches()

        eddl.reset_loss(model)

        pbar = tqdm(range(1, n_train_batches+1))
        pbar.set_description("Training")
        for batch in pbar:
            # Load batch of data
            dataset.LoadBatch(x, y)
            # Perform training with batch: forward and backward
            eddl.train_batch(model, [x], [y])
            # Get current metrics
            losses = eddl.get_losses(model)
            metrics = eddl.get_metrics(model)
            # Log in the progress bar
            pbar.set_postfix({"loss": losses[0], "acc": metrics[0]})

        ####################
        # Validation phase #
        ####################

        # Prepare dataset
        dataset.SetSplit(ecvl.SplitType.validation)

        eddl.reset_loss(model)

        pbar = tqdm(range(1, n_val_batches+1))
        pbar.set_description("Validation")
        for batch in pbar:
            # Load batch of data
            dataset.LoadBatch(x, y)
            # Perform forward computations
            eddl.eval_batch(model, [x], [y])
            # Get current metrics
            losses = eddl.get_losses(model)
            metrics = eddl.get_metrics(model)
            # Log in the progress bar
            pbar.set_postfix({"val_loss": losses[0], "val_acc": metrics[0]})

        if metrics[0] > best_acc:
            best_acc = metrics[0]
            model_name = f"{exp_name}_epoch-{epoch}_acc-{best_acc:.4f}.onnx"
            model_path = os.path.join(args.models_ckpts, model_name)
            print(f"New best model! Saving ONNX to: {model_path}")
            eddl.save_net_to_onnx_file(model, model_path)

        return []  # TODO History list
