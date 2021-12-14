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

    # Filter samples that match the filepath with the regex filter
    if args.regex:
        print(f"Going to filter the samples with the regex '{args.regex}'")
        print(f"Samples before filtering: {len(full_df.index)}")
        full_df = full_df[full_df.filepath.str.contains(args.regex)]
        print(f"Samples after filtering: {len(full_df.index)}")

    labels_not_used = set(args.all_labels).difference(set(args.target_labels))

    # Prepare the list of labels for each sample
    def create_labels_list(row):
        sample_labels = []
        for label in args.target_labels:
            if row[label] == 1:
                sample_labels.append(label)

        if args.add_normal and len(sample_labels) == 0:
            found = False
            for label in labels_not_used:
                if row[label] == 1:
                    found = True
            if not found:
                sample_labels = ['normal']

        return sample_labels

    full_df['labels'] = full_df.apply(create_labels_list, axis=1)

    if args.add_normal:
        before_drop = len(full_df.index)
        full_df = full_df[full_df['labels'].map(lambda x: len(x)) > 0]
        samples_removed = before_drop - len(full_df.index)
        print(f"Samples without a target label removed: {samples_removed}")
        args.target_labels.append('normal')

    # Create the final YAML from the processed dataframe 'full_df'
    create_ecvl_yaml(full_df, args.yaml_path, args.target_labels, True)

    # Store the Dataframe to a TSV for the Pytorch pipeline
    if args.tsv_path:
        aux_df = full_df[['subject', 'session', 'filepath', 'labels',
                          'gender', 'age', 'split']]
        aux_df.to_csv(args.tsv_path, sep='\t', index=False)


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

    arg_parser.add_argument("--tsv-path",
                            help=("Path to the store the new TSV file "
                                  "(for the Pytorch pipeline)"),
                            required=False,
                            default="",
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

    arg_parser.add_argument("--all-labels",
                            help=("List with all the labels present in the "
                                  "dataset CSV files provided"),
                            metavar="label_name",
                            nargs='+',
                            default=["infiltrates", "pneumonia", "covid19"],
                            type=str)

    arg_parser.add_argument("--add-normal",
                            help=("The samples without any label set to 1 are"
                                  " labeled as 'normal'"),
                            action="store_true")

    arg_parser.add_argument("--regex",
                            help="Regular expresion to filter the samples by filepath",
                            default="",
                            type=str)

    main(arg_parser.parse_args())
