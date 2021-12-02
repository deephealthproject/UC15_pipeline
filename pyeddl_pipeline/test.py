"""
Main script to train the models.
"""
import os
import argparse
import json

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl

from lib.training import get_augmentations, test


def main(args):

    ###################
    # Prepare Dataset #
    ###################

    # Prepare data augmentations for the splits (train, validation, test)
    splits_augs = get_augmentations("0.0", args.target_size)

    # Create the dataset object to generate the batches of data
    color_type = ecvl.ColorType.RGB if args.rgb else ecvl.ColorType.GRAY
    dataset = ecvl.DLDataset(args.yaml_path,
                             args.batch_size,
                             splits_augs,
                             color_type)

    ###################
    # Perform testing #
    ###################

    for onnx_path in args.onnx_files:
        print(f"\nGoing to load the model \"{onnx_path}\" for testing")
        model = eddl.import_net_from_onnx_file(onnx_path)

        # Get the computing device
        if args.cpu:
            comp_serv = eddl.CS_CPU(-1, args.mem_level)
        else:
            comp_serv = eddl.CS_GPU(args.gpus, 1, args.mem_level)

        eddl.build(model,
                   eddl.adam(0.0001),
                   ["softmax_cross_entropy"],
                   ["accuracy"],
                   comp_serv,
                   False)  # Avoid weights initialization

        # Inference on test split
        test_results = test(model,
                            dataset,
                            args,
                            store_results=False)

        # Show tests results
        print(f"\nTest results of '{onnx_path}':")
        print(f"  - loss={test_results['loss']:.4f}")
        print(f"  - acc={test_results['acc']:.4f}")
        print("\nTest report:")
        print(json.dumps(test_results['report'], indent=4))

        if args.out_path:
            fname = f"test_res_{os.path.basename(onnx_path)[:-5]}.json"
            with open(os.path.join(args.out_path, fname), 'w') as fstream:
                json.dump(test_results, fstream, indent=4, sort_keys=True)

        del model  # Free the memory for the next model


if __name__ == "__main__":
    # Get the config from the script arguments
    arg_parser = argparse.ArgumentParser(
        description="Script to perform test with one or more models (with ensemble)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        "--yaml-path",
        help="Path to the YAML file to create the ECVL dataset",
        default="../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.yaml")

    arg_parser.add_argument(
        "--multiclass",
        help="Prepares the testing pipeline for multiclass classification",
        action="store_true")

    arg_parser.add_argument(
        "--binary-loss",
        help=("Prepares the pipeline for models with as many output layers as"
              " classes (with one output neuron)"),
        action="store_true")

    arg_parser.add_argument(
        "--onnx-files",
        help="A list of paths to the ONNX files to use for testing",
        metavar="onnx_path",
        nargs='+',
        default=[],
        type=str)

    arg_parser.add_argument(
        "--out-path",
        help=("Path of the folder to store the test results. If not provided, "
              "the tests are only shown in the standard output"),
        default="")

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
        "--rgb",
        help="Load the images in RGB format instead of grayscale",
        action="store_true")

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
        "--seed",
        help="Seed value for random operations",
        default=1234,
        type=int)

    main(arg_parser.parse_args())
