import argparse
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from lib.training import get_augmentations


def main(args):
    splits_augs = get_augmentations(args.augmentations, args.target_size)

    dataset = ecvl.DLDataset(args.yaml_path,
                             args.batch_size,
                             splits_augs,
                             ecvl.ColorType.GRAY)

    in_shape = (dataset.n_channels_, *args.target_size)
    num_classes = len(dataset.classes_)
    print("Data INFO:")
    print(f" - num classes: {num_classes}")
    print(f" - input shape: {in_shape}")

    x = Tensor([args.batch_size, *in_shape])
    y = Tensor([args.batch_size, num_classes])

    dataset.SetSplit(ecvl.SplitType.training)

    dataset.LoadBatch(x, y)

    print("\nBatch data:")
    print("Images:")
    x.info()
    print(f"mean: {x.mean()} - max: {x.max()} - min: {x.min()}")
    x.print()

    print("\nLabels:")
    y.info()
    y.print()


if __name__ == "__main__":
    # Get the config from the script arguments
    arg_parser = argparse.ArgumentParser(
        description="Script for playing with the ECVL Dataset object",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        "--yaml-path",
        help="Path to the YAML file to create the ECVL dataset",
        default="../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.yaml")

    arg_parser.add_argument(
        "--augmentations", "-augs",
        help="Set of augmentations to select",
        default="0.0",
        choices=["0.0", "1.0", "1.1"],
        type=str)

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
        type=int,
        default=1)

    main(arg_parser.parse_args())
