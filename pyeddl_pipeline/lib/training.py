"""
Module with utility functions for the training pipeline.
"""
import os
import argparse
import random
import time
import json
import statistics

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl


#########
# Utils #
#########

def is_normal_vs_classification(classes: list) -> bool:
    """
    Checks if the classes provides correspond to a binary classification task
    of 'normal' vs 'other_class'.

    Args:
        classes: List of class names

    Returns:
        The index of the 'other_class' string in the list provided. Returns -1
        if the task is not 'normal' vs 'other_class'
    """
    if len(classes) != 2:
        return -1

    if classes[0] == 'normal':
        return 1
    if classes[1] == 'normal':
        return 0

    return -1


def get_losses_and_metrics(args: argparse.Namespace) -> tuple:
    """
    Creates the vectors of losses and metrics to build the model.

    Args:
        args: The argparse object with all the configuration data like:
              batch_size, epochs...

    Returns:
        A tuple (losses, metrics) where:
            - losses: Vector of strings with the name of the losses
            - metrics: Vector of strings with the name of the metrics
    """
    if args.binary_loss:
        losses = ["binary_cross_entropy"]
    elif args.multiclass:
        losses = ["mse"]
    else:
        losses = ["softmax_cross_entropy"]

    if args.normal_vs_classification:
        metrics = ['binary_accuracy']
    else:
        metrics = ['accuracy']

    return (losses, metrics)


#####################
# Data Augmentation #
#####################


def augmentations_v2_0(size: tuple) -> ecvl.DatasetAugmentations:
    """
    Returns the v2.0 augmentations for each split (train, val, test).
    The v2.0 applies the same augmentations types than the v1.X and adds
    optical distorsion.
    """
    tr_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugOpticalDistortion([-0.2, -0.2], [-0.2, -0.2],
                                  interp=ecvl.InterpolationType.cubic),
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugMirror(p=0.5),
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


def augmentations_v1_0(size: tuple) -> ecvl.DatasetAugmentations:
    """
    Returns the v1.0 augmentations for each split (train, val, test).
    The v1.0 applies some basic augmentations playing with the brightness,
    contrast and small image rotations.
    """
    tr_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size, ecvl.InterpolationType.cubic),
        ecvl.AugMirror(p=0.5),
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
        ecvl.AugMirror(p=0.5),
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
    if version == "2.0":
        return augmentations_v2_0(size)

    raise Exception("Wrong augmentations version provided!")


##################
# Regularization #
##################


def apply_regularization(model: eddl.Model,
                         regularization: str,
                         factor: float):
    """
    Applies regularization to all the layers of the model.

    Args:
        model: Model to apply the regularization.

        regularization: Name of the regularization type to apply.

        factor: Regularization factor to use.
    """
    # Get the regularization function to apply
    if regularization == "l1":
        reg_func = lambda l: eddl.L1(l, factor)
    elif regularization == "l2":
        reg_func = lambda l: eddl.L2(l, factor)
    elif regularization == "l1l2":
        reg_func = lambda l: eddl.L1L2(l, factor, factor)
    else:
        raise Exception(f"The regularization type provided '{regularization}' is not valid!")

    # Set the regularization for each layer of the model
    for layer in model.layers:
        reg_func(layer)


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
            - best_model_byloss: Path to the ONNX file with the best model
                                 selected by val_loss
            - best_model_byacc: Path to the ONNX file with the best model
                                selected by val_acc
    """
    if args.normal_vs_classification:
        class_idx = is_normal_vs_classification(dataset.classes_)
        if class_idx == -1:
            raise Exception("The classes of the dataset are not valid "
                            "for binary classification (normal vs OTHER)")

    n_train_samples = len(dataset.GetSplit(ecvl.SplitType.training))
    n_train_batches = n_train_samples // args.batch_size
    n_val_batches = dataset.GetNumBatches(ecvl.SplitType.validation)

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

    avoid_unfreeze = False  # To avoid unfreezing the weights more than once

    print(f"Going to train for {args.epochs} epochs:")
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}:")
        ##################
        # Training phase #
        ##################

        if not avoid_unfreeze:
            # Check if the model must be unfreezed
            if args.is_pretrained and 0 < args.frozen_epochs < epoch:
                print("Going to unfreeze the pretrained weights")
                for layer_name in args.pretrained_layers:
                    eddl.setTrainable(model, layer_name, True)
                avoid_unfreeze = True

        # Update learning rate
        new_lr = args.learning_rate * (1 / (1 + args.lr_decay * (epoch-1)))
        eddl.setlr(model, [new_lr])
        print(f"Current learning rate: {new_lr}")

        # Prepare dataset
        dataset.SetSplit(ecvl.SplitType.training)
        dataset.ResetBatch(ecvl.SplitType.training, shuffle=True)
        dataset.Start()  # Spawn workers

        eddl.reset_loss(model)

        pbar = tqdm(range(1, n_train_batches+1))
        # To track the average load and training time for each batch
        load_time = 0
        train_time = 0
        for batch in pbar:
            # Load batch of data
            load_start = time.perf_counter()
            vsamples, x, y = dataset.GetBatch()
            load_end = time.perf_counter()
            load_time += load_end - load_start
            if args.normal_vs_classification:
                # Select the value for the label: {0: normal, 1: other_class}
                y = y.select([":", f"{class_idx}"])
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
                              "data_queue": f"{dataset.GetQueueSize()}",
                              "avg_train_time": f"{train_time / batch:.3f}s"})

        # Save the epoch results of the train split
        history["loss"].append(losses[0])
        history["acc"].append(metrics[0])

        dataset.Stop()  # Join worker threads

        ####################
        # Validation phase #
        ####################

        # Prepare dataset
        dataset.SetSplit(ecvl.SplitType.validation)
        dataset.ResetBatch(ecvl.SplitType.validation, shuffle=False)
        dataset.Start()  # Spawn workers

        eddl.reset_loss(model)

        pbar = tqdm(range(1, n_val_batches+1))
        # To track the average load and evaluation time for each batch
        load_time = 0
        eval_time = 0
        for batch in pbar:
            # Load batch of data
            load_start = time.perf_counter()
            vsamples, x, y = dataset.GetBatch()
            load_end = time.perf_counter()
            load_time += load_end - load_start
            if args.normal_vs_classification:
                # Select the value for the label: {0: normal, 1: other_class}
                y = y.select([":", f"{class_idx}"])
            model.resize(x.shape[0])
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
                              "data_queue": f"{dataset.GetQueueSize()}",
                              "avg_eval_time": f"{eval_time / batch:.3f}s"})

        # Save the epoch results of the validation split
        history["val_loss"].append(losses[0])
        history["val_acc"].append(metrics[0])

        epoch_loss = losses[0]
        epoch_acc = metrics[0]

        # Save the best models by val_loss and val_acc to ONNX
        new_best_loss = epoch_loss < best_loss
        new_best_acc = epoch_acc > best_acc
        if new_best_loss and new_best_acc:
            best_loss = epoch_loss
            best_acc = epoch_acc
            model_name = (f"{exp_name}_epoch-{epoch}_"
                          f"loss-{best_loss:.4f}_acc-{best_acc:.4f}_"
                          "by-loss-and-acc.onnx")
            model_path = os.path.join(exp_ckpts_path, model_name)
            print(("New best model by val_loss and val_acc! "
                  f"Saving ONNX to: {model_path}"))
            eddl.save_net_to_onnx_file(model, model_path)
            history["best_model_byloss"] = model_path
            history["best_model_byacc"] = model_path
        elif new_best_acc:
            aux_loss = epoch_loss
            best_acc = epoch_acc
            model_name = (f"{exp_name}_epoch-{epoch}_"
                          f"loss-{aux_loss:.4f}_acc-{best_acc:.4f}_"
                          "by-acc.onnx")
            model_path = os.path.join(exp_ckpts_path, model_name)
            print(f"New best model by val_acc! Saving ONNX to: {model_path}")
            eddl.save_net_to_onnx_file(model, model_path)
            history["best_model_byacc"] = model_path
        elif new_best_loss:
            best_loss = epoch_loss
            aux_acc = epoch_acc
            model_name = (f"{exp_name}_epoch-{epoch}_"
                          f"loss-{best_loss:.4f}_acc-{aux_acc:.4f}_"
                          "by-loss.onnx")
            model_path = os.path.join(exp_ckpts_path, model_name)
            print(f"New best model by val_loss! Saving ONNX to: {model_path}")
            eddl.save_net_to_onnx_file(model, model_path)
            history["best_model_byloss"] = model_path

        # Get the epoch metrics
        epoch_res = {metric: history[metric][-1] for metric in metrics_names}
        epoch_res["epoch"] = epoch
        # Update the results CSV
        results_df = results_df.append(epoch_res, ignore_index=True)
        results_df.to_csv(results_csv_path, index=False)

        dataset.Stop()  # Join worker threads

    return history


def test(model: eddl.Model,
         dataset: ecvl.DLDataset,
         args: argparse.Namespace,
         out_filename: str = "test_res.json",
         store_results: bool = True) -> dict:
    """
    Performs the model evaluation with the test split.

    Args:
        model: Trained EDDL model to test.

        dataset: ECVL dataset to load the data (from test split).

        args: The argparse object with all the configuration data like:
              batch_size, epochs...

        out_filename: Name of the JSON file to create inside the experiment
                      folder with the test results

        store_results: If True stores the test results in the experiment folder

    Returns:
        A dictionary with a summary of the testing phase.
        keys:
            - loss: Test loss
            - acc: Test accuracy
            - report: Full report with more metrics by class
                      (Check: sklearn.metrics.classification_report)
    """
    if args.normal_vs_classification:
        class_idx = is_normal_vs_classification(dataset.classes_)
        if class_idx == -1:
            raise Exception("The classes of the dataset are not valid "
                            "for binary classification (normal vs OTHER)")

    n_test_batches = dataset.GetNumBatches(ecvl.SplitType.test)

    # To store and return the testing results
    metrics_names = ["loss", "acc"]
    history = {metric: -1 for metric in metrics_names}

    # Prepare dataset
    dataset.SetSplit(ecvl.SplitType.test)
    dataset.ResetBatch(ecvl.SplitType.test, shuffle=False)
    dataset.Start()  # Spawn workers

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
        vsamples, x, y = dataset.GetBatch()
        load_end = time.perf_counter()
        load_time += load_end - load_start
        if args.normal_vs_classification:
            # Select the value for the label: {0: normal, 1: other_class}
            y = y.select([":", f"{class_idx}"])
        model.resize(x.shape[0])
        # Perform forward computations
        test_start = time.perf_counter()
        eddl.eval_batch(model, [x], [y])
        test_end = time.perf_counter()
        test_time += test_end - test_start
        # Store the predictions to compute statistics later
        batch_logits = eddl.getOutput(eddl.getOut(model)[0]).getdata()
        if args.multiclass or args.binary_loss:
            batch_preds = np.where(batch_logits > 0.5, 1, 0)
            batch_targets = y.getdata()
        else:
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
                          "data_queue": f"{dataset.GetQueueSize()}",
                          "avg_test_time": f"{test_time / batch:.3f}s"})

    dataset.Stop()  # Join worker threads

    if args.normal_vs_classification:
        # Ensure that the class 0 is normal
        classes_names = ['normal', dataset.classes_[class_idx]]
    else:
        classes_names = dataset.classes_

    # Compute a report with statistics for each target class
    report = classification_report(targets,
                                   preds,
                                   target_names=classes_names,
                                   output_dict=True)
    sklearn_acc = accuracy_score(targets, preds)

    binary_multilabel = args.binary_loss and not args.normal_vs_classification
    if not args.multiclass and not binary_multilabel:
        balanced_acc = balanced_accuracy_score(targets, preds)
        history["balanced_acc"] = balanced_acc
    else:
        multilabel_confmat = multilabel_confusion_matrix(targets, preds)
        history["multilabel_confmat"] = str(multilabel_confmat)

    # Save test results
    history["loss"] = losses[0]
    history["acc"] = metrics[0]
    history["sklearn_acc"] = sklearn_acc
    history["report"] = report

    # Save the tests results in a JSON file
    if store_results:
        with open(os.path.join(args.exp_path, out_filename), 'w') as fstream:
            json.dump(history, fstream, indent=4, sort_keys=True)

    return history
