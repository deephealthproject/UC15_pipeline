"""
Script for training the models using Pytorch.
"""
import os
import argparse
import multiprocessing
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lib.data import COVIDDataModule
from lib.models import get_model


def main(args):
    seed_everything(args.seed, workers=True)  # For reproducibility

    # Experiment name
    exp_strftime = datetime.now().strftime("%d-%b_%H:%M")
    exp_name = (f"{exp_strftime}"
                f"_net-{args.model}"
                f"_DA-{args.augmentations}"
                f"_input-{args.target_size[0]}x{args.target_size[1]}"
                f"_opt-{args.optimizer}"
                f"_lr-{args.learning_rate}")

    # Prepare the folder to store the experiment results
    exp_path = os.path.join(args.logs, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    # Create the object to handle the data loading
    data_module = COVIDDataModule(args.data_tsv,
                                  args.labels,
                                  args.batch_size,
                                  shuffle=True,
                                  target_size=args.target_size,
                                  augmentations=args.augmentations,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  to_rgb=args.to_rgb)

    # Create the model
    model = get_model(args.model,
                      (1, *args.target_size),  # Input shape
                      len(args.labels),  # Number of classes
                      args)

    # Prepare training callbacks
    exp_ckpts_path = os.path.join(exp_path, "ckpts")
    ckpt_callback = ModelCheckpoint(dirpath=exp_ckpts_path,
                                    monitor="val_loss",
                                    save_top_k=1)

    # Prepare the logger
    logger = TensorBoardLogger(exp_path, name="tensorboard_logs")

    # Create the object to configure and execute training
    trainer = Trainer(gpus=args.gpus,
                      deterministic=True,  # For reproducibility
                      callbacks=[ckpt_callback],
                      max_epochs=args.epochs,
                      profiler=args.profiler,
                      logger=logger)

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test with test split
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    # Get the config from the script arguments
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        "--data-tsv",
        help="Path to the TSV file to create the data loaders",
        default="../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.tsv")

    arg_parser.add_argument(
        "--target-size",
        help="Target size to resize the images, given by height and width",
        metavar=("HEIGHT", "WIDTH"),
        nargs=2,
        default=[256, 256],
        type=int)

    arg_parser.add_argument(
        "--batch-size", "-bs",
        help="Size of the training batches of data",
        default=16)

    arg_parser.add_argument(
        "--labels",
        help=("Target labels to select the samples for training. "
              "The order is important, the first matching label will be taken "
              "as the label for the sample."),
        metavar="label_name",
        nargs='+',
        default=["normal", "COVID 19", "pneumonia", "infiltrates"],
        type=str)

    arg_parser.add_argument(
        "--epochs",
        help="Number of epochs to train",
        default=10,
        type=int)

    arg_parser.add_argument(
        "--model",
        help="Model architecture to train",
        default="ResNet18",
        type=str,
        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                 "PretrainedResNet18", "PretrainedResNet34",
                 "PretrainedResNet50", "PretrainedResNet101",
                 "PretrainedResNet152"])

    arg_parser.add_argument(
        "--optimizer", "-opt",
        help="Training optimizer",
        default="Adam",
        choices=["Adam", "SGD"],
        type=str)

    arg_parser.add_argument(
        "--learning-rate", "-lr",
        help="Learning rate of the optimizer",
        default=0.001,
        type=float)

    arg_parser.add_argument(
        "--augmentations", "-augs",
        help="Set of augmentations to select",
        default="0.0",
        choices=["0.0", "1.0", "1.1"],
        type=str)

    arg_parser.add_argument(
        "--to-rgb",
        help=("Converts grayscale images to rgb replicating the single channel"
              " (Used for pretrained models)"),
        action="store_true")

    arg_parser.add_argument(
        "--gpus",
        help="Number of gpus to use during training",
        default=1,
        type=int)

    arg_parser.add_argument(
        "--num-workers",
        help="Number of processes to load the data.",
        default=multiprocessing.cpu_count(),
        type=int)

    arg_parser.add_argument(
        "--seed",
        help="Seed value for random operations",
        default=1234,
        type=int)

    arg_parser.add_argument(
        "--profiler",
        help="Selects a performance profiler to test the pipeline",
        choices=["simple", "advanced"],
        default=None)

    arg_parser.add_argument(
        "--logs",
        help="Path to the folder for saving the experiments logs",
        default="logs")

    main(arg_parser.parse_args())
