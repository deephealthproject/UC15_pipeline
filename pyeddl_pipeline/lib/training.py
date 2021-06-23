"""
Module with utility functions for the training pipeline.
"""
import os
import argparse
import random
import time
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

#####################
# Data Augmentation #
#####################


def augmentations_v1_0(size: tuple) -> ecvl.DatasetAugmentations:
    """
    Returns the v1.0 augmentations for each split (train, val, test).
    The v1.0 applies some basic augmentations playing with the brightness,
    contrast and small image rotations.
    """
    tr_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugRotate([-10, 10]),
        ecvl.AugBrightness([0, 50]),
        ecvl.AugGammaContrast([0.8, 1.2]),
        ecvl.AugToFloat32(255.0)
    ])

    val_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugToFloat32(255.0)
    ])

    te_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugToFloat32(255.0)
    ])

    return ecvl.DatasetAugmentations([tr_augs, val_augs, te_augs])


def augmentations_v1_1(size: tuple) -> ecvl.DatasetAugmentations:
    """
    Returns the v1.1 augmentations for each split (train, val, test).
    The v1.1 applies the same augmentations than the v1.0 but more
    aggressively.
    """
    tr_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugRotate([-15, 15]),
        ecvl.AugBrightness([0, 70]),
        ecvl.AugGammaContrast([0.6, 1.4]),
        ecvl.AugToFloat32(255.0)
    ])

    val_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugToFloat32(255.0)
    ])

    te_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugToFloat32(255.0)
    ])

    return ecvl.DatasetAugmentations([tr_augs, val_augs, te_augs])


def augmentations_v0_0(size: tuple) -> ecvl.DatasetAugmentations:
    """
    Returns the v0.0 augmentations for each split (train, val, test).
    The v0.0 is just for resizing the data, not to perform data augmentation.
    """
    tr_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugToFloat32(255.0)
    ])

    val_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugToFloat32(255.0)
    ])

    te_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugToFloat32(255.0)
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
    if version == "0.0":
        return augmentations_v0_0(size)
    if version == "1.0":
        return augmentations_v1_0(size)
    if version == "1.1":
        return augmentations_v1_1(size)

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
          exp_name: str,
          args: argparse.Namespace) -> list:
    """
    Executes the main training loop. Performs training, validation and model
    checkpoints.

    Args:
        model: EDDL model to train. Must be already built.

        dataset: ECVL dataset to load the data for training.

        exp_name: Name of the training experiment.

        args: The argparse object with all the configuration data like:
              batch_size, epochs...

    Returns:
        A dictionary with a summary of the training phase.
        keys:
            - loss: Loss of each epoch with the training split
            - acc: Accuracy of each epoch with the training split
            - val_loss: Loss of each epoch with the validation split
            - val_acc: Accuracy of each epoch with the validation split
            - best_model: Path to the ONNX file with the best model
    """
    n_train_samples = len(dataset.GetSplit(ecvl.SplitType.training))
    n_train_batches = n_train_samples // args.batch_size
    n_val_samples = len(dataset.GetSplit(ecvl.SplitType.validation))
    n_val_batches = n_val_samples // args.batch_size

    # Create auxiliary tensors to load the data
    x = Tensor([args.batch_size, *args.in_shape])  # Images
    y = Tensor([args.batch_size, args.num_classes])  # Labels

    # To store and return the training results
    metrics_names = ["loss", "acc", "val_loss", "val_acc"]
    history = {metric: [] for metric in metrics_names}
    best_loss = float('inf')  # To track the best model
    best_acc = 0.0

    # Save the experiment config in a JSON file
    with open(os.path.join(args.exp_path, "args.json"), 'w') as fstream:
        json.dump(vars(args), fstream, indent=4, sort_keys=True)

    # Prepare the checkpoints folder
    exp_ckpts_path = os.path.join(args.exp_path, "ckpts")
    os.makedirs(exp_ckpts_path, exist_ok=True)

    # Prepare a CSV to store the training metrics for each epoch
    results_df = pd.DataFrame(columns=["epoch"] + metrics_names)
    # Path to store the results CSV
    results_csv_path = os.path.join(args.exp_path, "train_res.csv")

    random.seed(args.seed)  # Seed for shuffling the data

    print(f"Going to train for {args.epochs} epochs:")
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}:")
        ##################
        # Training phase #
        ##################

        # Check if we have to unfreeze the weights
        if args.is_pretrained and 0 < args.frozen_epochs < epoch:
            print("Going to unfreeze the pretrained weights")
            for layer_name in args.pretrained_layers:
                eddl.setTrainable(model, layer_name, True)

        # Prepare dataset
        dataset.SetSplit(ecvl.SplitType.training)
        shuffle_training_split(dataset)
        dataset.ResetAllBatches()

        eddl.reset_loss(model)

        pbar = tqdm(range(1, n_train_batches+1))
        # To track the average load and training time for each batch
        load_time = 0
        train_time = 0
        for batch in pbar:
            # Load batch of data
            load_start = time.perf_counter()
            dataset.LoadBatch(x, y)
            load_end = time.perf_counter()
            load_time += load_end - load_start
            # Perform training with batch: forward and backward
            train_start = time.perf_counter()
            eddl.train_batch(model, [x], [y])
            train_end = time.perf_counter()
            train_time += train_end - train_start
            # Get current metrics
            losses = eddl.get_losses(model)
            metrics = eddl.get_metrics(model)
            # Log in the progress bar
            pbar.set_description(
                f"Training[loss={losses[0]:.4f}, acc={metrics[0]:.4f}]")
            pbar.set_postfix({"avg_load_time": f"{load_time / batch:.3f}s",
                              "avg_train_time": f"{train_time / batch:.3f}s"})

        # Save the epoch results of the train split
        history["loss"].append(losses[0])
        history["acc"].append(metrics[0])

        ####################
        # Validation phase #
        ####################

        # Prepare dataset
        dataset.SetSplit(ecvl.SplitType.validation)

        eddl.reset_loss(model)

        pbar = tqdm(range(1, n_val_batches+1))
        # To track the average load and evaluation time for each batch
        load_time = 0
        eval_time = 0
        for batch in pbar:
            # Load batch of data
            load_start = time.perf_counter()
            dataset.LoadBatch(x, y)
            load_end = time.perf_counter()
            load_time += load_end - load_start
            # Perform forward computations
            eval_start = time.perf_counter()
            eddl.eval_batch(model, [x], [y])
            eval_end = time.perf_counter()
            eval_time += eval_end - eval_start
            # Get current metrics
            losses = eddl.get_losses(model)
            metrics = eddl.get_metrics(model)
            # Log in the progress bar
            pbar.set_description(
                f"Validation[val_loss={losses[0]:.4f}, val_acc={metrics[0]:.4f}]")
            pbar.set_postfix({"avg_load_time": f"{load_time / batch:.3f}s",
                              "avg_eval_time": f"{eval_time / batch:.3f}s"})

        # Save the epoch results of the validation split
        history["val_loss"].append(losses[0])
        history["val_acc"].append(metrics[0])

        if losses[0] < best_loss:
            best_loss = losses[0]
            best_acc = metrics[0]
            model_name = (f"{exp_name}_epoch-{epoch}_"
                          f"loss-{best_loss:.4f}_acc-{best_acc:.4f}.onnx")
            model_path = os.path.join(exp_ckpts_path, model_name)
            print(f"New best model! Saving ONNX to: {model_path}")
            eddl.save_net_to_onnx_file(model, model_path)
            history["best_model"] = model_path

        # Get the epoch metrics
        epoch_res = {metric: history[metric][-1] for metric in metrics_names}
        epoch_res["epoch"] = epoch
        # Update the results CSV
        results_df = results_df.append(epoch_res, ignore_index=True)
        results_df.to_csv(results_csv_path, index=False)

    return history


def test(model: eddl.Model,
         dataset: ecvl.DLDataset,
         args: argparse.Namespace) -> list:
    """
    Performs the model evaluation with the test split.

    Args:
        model: Trained EDDL model to test.

        dataset: ECVL dataset to load the data (from test split).

        args: The argparse object with all the configuration data like:
              batch_size, epochs...

    Returns:
        A dictionary with a summary of the testing phase.
        keys:
            - loss: Test loss
            - acc: Test accuracy
            - report: Full report with more metrics by class
                      (Check: sklearn.metrics.classification_report)
    """
    n_test_samples = len(dataset.GetSplit(ecvl.SplitType.test))
    n_test_batches = n_test_samples // args.batch_size

    # Create auxiliary tensors to load the data
    x = Tensor([args.batch_size, *args.in_shape])  # Images
    y = Tensor([args.batch_size, args.num_classes])  # Labels

    # To store and return the testing results
    history = {"loss": -1, "acc": -1}

    # Prepare dataset
    dataset.SetSplit(ecvl.SplitType.test)
    dataset.ResetAllBatches()

    eddl.reset_loss(model)

    print("Testing:")
    pbar = tqdm(range(1, n_test_batches+1))
    # To track the average load and testing time for each batch
    load_time = 0
    test_time = 0
    # To store the predictions and labels to compute statistics
    preds = None
    targets = None
    for batch in pbar:
        # Load batch of data
        load_start = time.perf_counter()
        dataset.LoadBatch(x, y)
        load_end = time.perf_counter()
        load_time += load_end - load_start
        # Perform forward computations
        test_start = time.perf_counter()
        eddl.eval_batch(model, [x], [y])
        test_end = time.perf_counter()
        test_time += test_end - test_start
        # Store the predictions to compute statistics later
        batch_logits = eddl.getOutput(eddl.getOut(model)[0]).getdata()
        batch_preds = np.argmax(batch_logits, axis=1)
        batch_targets = np.argmax(y.getdata(), axis=1)
        if preds is None:
            preds = batch_preds
            targets = batch_targets
        else:
            preds = np.concatenate((preds, batch_preds))
            targets = np.concatenate((targets, batch_targets))
        # Get current metrics
        losses = eddl.get_losses(model)
        metrics = eddl.get_metrics(model)
        # Log in the progress bar
        pbar.set_description(
            f"Test[loss={losses[0]:.4f}, acc={metrics[0]:.4f}]")
        pbar.set_postfix({"avg_load_time": f"{load_time / batch:.3f}s",
                          "avg_test_time": f"{test_time / batch:.3f}s"})

    # Compute a report with statistics for each target class
    report = classification_report(targets,
                                   preds,
                                   target_names=dataset.classes_,
                                   output_dict=True)

    # Save test results
    history["loss"] = losses[0]
    history["acc"] = metrics[0]
    history["report"] = report

    # Save the tests results in a JSON file
    with open(os.path.join(args.exp_path, "test_res.json"), 'w') as fstream:
        json.dump(history, fstream, indent=4, sort_keys=True)

    return history
