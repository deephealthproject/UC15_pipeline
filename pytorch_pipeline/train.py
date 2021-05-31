"""
Script for training the models using Pytorch.
"""
import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.data import COVIDDataModule


def main(args):
    seed_everything(args.seed, workers=True)  # For reproducibility

    # Create the object to handle the data loading
    data_module = COVIDDataModule(args.data_tsv,
                                  args.labels,
                                  args.batch_size,
                                  shuffle=True,
                                  target_size=args.target_size,
                                  augmentations=args.augmentations)

    # Prepare training callbacks
    ckpt_callback = ModelCheckpoint(dirpath=args.models_ckpts,
                                    monitor="val_loss",
                                    save_top_k=1)

    # Create the object to configure and execute training
    trainer = Trainer(gpus=args.gpus,
                      deterministic=True,  # For reproducibility
                      callbacks=[ckpt_callback])


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
        "--augmentations", "-augs",
        help="Set of augmentations to select",
        default="0.0",
        choices=["0.0"],
        type=str)

    arg_parser.add_argument(
        "--gpus",
        help="Number of gpus to use during training",
        default=1,
        type=int)

    arg_parser.add_argument(
        "--seed",
        help="Seed value for random operations",
        default=1234,
        type=int)

    arg_parser.add_argument(
        "--models-ckpts",
        help="Path to the folder for saving the ONNX models checkpoints",
        default="models_ckpts")

    main(arg_parser.parse_args())
