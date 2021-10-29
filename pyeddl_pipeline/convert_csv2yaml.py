import os
import argparse

import pandas as pd

from lib.data_processing import create_ecvl_yaml


def main(args):
    # Load the CSV data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    # Prepare the filepaths for the YAML
    if args.paths_prefix != "":
        def path_fix_func(path):
            return os.path.join(args.paths_prefix, path)

        train_df.filepath = train_df.filepath.apply(path_fix_func)
        val_df.filepath = val_df.filepath.apply(path_fix_func)
        test_df.filepath = test_df.filepath.apply(path_fix_func)

    # Add the column with the corresponding training split
    train_df['split'] = 'training'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'

    # Join the dataframes
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Prepare the list of labels for each sample
    def create_labels_list(row):
        sample_labels = []
        for label in args.target_labels:
            if row[label] == 1:
                sample_labels.append(label)

        return sample_labels

    full_df['labels'] = full_df.apply(create_labels_list, axis=1)

    # Create the final YAML from the processed dataframe 'full_df'
    create_ecvl_yaml(full_df, args.yaml_path, args.target_labels, True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description=("Script for converting the CSVs with the training "
                     "splits definition to a YAML file for the ECVL dataset"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument("--train-csv",
                            help="Path to the CSV of the train split",
                            required=True,
                            type=str)

    arg_parser.add_argument("--val-csv",
                            help="Path to the CSV of the validation split",
                            required=True,
                            type=str)

    arg_parser.add_argument("--test-csv",
                            help="Path to the CSV of the test split",
                            required=True,
                            type=str)

    arg_parser.add_argument("--yaml-path",
                            help="Path to the store the new YAML file",
                            required=True,
                            type=str)

    arg_parser.add_argument("--paths-prefix",
                            help="Prefix to add to the images paths",
                            default="",
                            type=str)

    arg_parser.add_argument("--target-labels",
                            help=("Labels to use for classification. The names"
                                  " should be the same in the CSVs columns"),
                            metavar="label_name",
                            nargs='+',
                            default=["infiltrates", "pneumonia", "covid19"],
                            type=str)

    main(arg_parser.parse_args())
