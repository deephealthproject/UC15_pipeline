"""
Main script to train the models.
"""
import os
import argparse
from datetime import datetime

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl

from lib.training import get_augmentations, get_optimizer, train, test
from lib.models import get_model
from lib.plot_utils import plot_training_results


def main(args):
    # Prepare data augmentations for the splits (train, validation, test)
    splits_augs = get_augmentations(args.augmentations, args.target_size)

    # Create the dataset object to generate the batches of data
    color_type = ecvl.ColorType.RGB if args.rgb else ecvl.ColorType.GRAY
    dataset = ecvl.DLDataset(args.yaml_path,
                             args.batch_size,
                             splits_augs,
                             color_type)

    # Data info
    in_shape = (dataset.n_channels_, *args.target_size)
    num_classes = len(dataset.classes_)
    args.in_shape = in_shape
    args.num_classes = num_classes

    if args.model_ckpt:
        # Load the model from an ONNX file
        model = eddl.import_net_from_onnx_file(args.model_ckpt)
        init_weights = False  # Avoid resetting the loaded weights
        l2init = []  # All the layers have weights from the ONNX file
    else:
        # Create the model
        model, init_weights, l2init = get_model(args.model,
                                                in_shape,
                                                num_classes)

    # Create the optimizer
    opt = get_optimizer(args.optimizer, args.learning_rate)

    # Get the computing device
    if args.cpu:
        comp_serv = eddl.CS_CPU(-1, args.mem_level)
    else:
        comp_serv = eddl.CS_GPU(args.gpus, 1, args.mem_level)

    # Build the model
    eddl.build(model,
               opt,
               ["softmax_cross_entropy"],
               ['accuracy'],
               comp_serv,
               init_weights)

    if not init_weights:
        # Initialize the new layers
        for layer_name in l2init:
            eddl.initializeLayer(model, layer_name)

    eddl.summary(model)  # Print the model layers

    # Experiment name
    exp_strftime = datetime.now().strftime("%d-%b_%H:%M")
    exp_name = (f"{exp_strftime}"
                f"_net-{args.model}"
                f"_DA-{args.augmentations}"
                f"_input-{args.target_size[0]}x{args.target_size[1]}"
                f"_opt-{args.optimizer}"
                f"_lr-{args.learning_rate}")

    if args.model_ckpt:
        # Get the name by taking the filename and removing the .onnx extension
        ckpt_name = os.path.basename(args.model_ckpt)[:-5]
        exp_name += f"_ckpt-{ckpt_name}"

    # Train the model
    history = train(model, dataset, exp_name, args)
    del model  # Free the memory before the testing phase

    # Create the plots of the training curves for loss and accuracy
    plot_training_results(history, exp_name, args.plots_path)

    # Load the best model f r testing
    print(f"\nGoing to load the model \"{history['best_model']}\" for testing")
    best_model = eddl.import_net_from_onnx_file(history["best_model"])

    # Create the optimizer
    opt = get_optimizer(args.optimizer, args.learning_rate)

    # Get the computing device
    if args.cpu:
        comp_serv = eddl.CS_CPU(-1, args.mem_level)
    else:
        comp_serv = eddl.CS_GPU(args.gpus, 1, args.mem_level)

    eddl.build(best_model,
               opt,
               ["softmax_cross_entropy"],
               ['accuracy'],
               comp_serv,
               False)  # Avoid weights initialization

    # Inference on test split
    test_results = test(best_model, dataset, args)
    test_loss = test_results['loss']
    test_acc = test_results['acc']
    test_report = test_results['report']
    print(f"\nTest results: loss={test_loss:.4f} - acc={test_acc:.4f}")
    print(f"Test report:\n{test_report}")


if __name__ == "__main__":
    # Get the config from the script arguments
    arg_parser = argparse.ArgumentParser(
        description="Script for training the classification models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        "--yaml-path",
        help="Path to the YAML file to create the ECVL dataset",
        default="../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.yaml")

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
        default=16)

    arg_parser.add_argument(
        "--epochs",
        help="Number of epochs to train",
        default=10,
        type=int)

    arg_parser.add_argument(
        "--augmentations", "-augs",
        help="Set of augmentations to select",
        default="0.0",
        choices=["0.0", "1.0", "1.1"],
        type=str)

    arg_parser.add_argument(
        "--rgb",
        help="Load the images in RGB format instead of grayscale",
        action="store_true")

    arg_parser.add_argument(
        "--model",
        help="Model architecture to train",
        default="model_1",
        type=str,
        choices=["model_1", "model_2", "model_3", "model_4",
                 "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                 "Pretrained_ResNet18", "Pretrained_ResNet34",
                 "Pretrained_ResNet50", "Pretrained_ResNet101",
                 "Pretrained_ResNet152"])

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
        "--cpu",
        help="Sets CPU as the computing device",
        action="store_true")

    arg_parser.add_argument(
        "--gpus",
        help='Sets the number of GPUs to use. Usage "--gpus 1 1" (two GPUs)',
        metavar="binary_flag",
        nargs="+",
        default=[1],
        type=int)

    arg_parser.add_argument(
        "--mem-level", "-mem",
        help="Memory level for the computing device",
        default="full_mem",
        choices=["full_mem", "mid_mem", "low_mem"],
        type=str)

    arg_parser.add_argument(
        "--models-ckpts",
        help="Path to the folder for saving the ONNX models checkpoints",
        default="models_ckpts")

    arg_parser.add_argument(
        "--plots-path",
        help="Path to the folder to store the training plots",
        default="plots")

    arg_parser.add_argument(
        "--seed",
        help="Seed value for random operations",
        default=1234,
        type=int)

    arg_parser.add_argument(
        "--model-ckpt",
        help="An ONNX model checkpoint to start training from",
        type=str)

    main(arg_parser.parse_args())
