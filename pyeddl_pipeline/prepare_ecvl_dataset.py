"""
This script prepares the YAML file needed by the ECVL library to create the
Dataset object to load the batches of data for training. This YAML file
contains the paths to the images and their corresponding metadata. The YAML
also provides the partitions for training, validation and test. Before
creating the YAML file the script also performs a first image preprocessing.
"""
import os
import argparse
import ast
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.data_processing import get_labels_from_str, create_ecvl_yaml
from lib.image_processing import histogram_equalization


def main(args):
    np.random.seed(args.seed)  # Set the random seed for Numpy

    assert sum(args.splits) == 1, "The splits values sum must be 1.0!"

    # Name of the directory to store the preprocessed images. It will be
    # created inside the subjects data folder ("covid19_posi").
    preproc_dirname = f"{args.yaml_name}_preproc_data"

    """
    Load and prepare data
    """
    derivatives_path = os.path.join(args.sub_path, "derivatives")  # Aux path

    # Load DataFrame with all the images (paths) by session and subject
    partitions_tsv_path = os.path.join(derivatives_path, "partitions.tsv")
    cols = ["subject", "session", "filepath"]  # Fix the original columns
    part_df = pd.read_csv(partitions_tsv_path, sep="\t", header=0, names=cols)

    # Load DataFrame with the labels for each session
    labels_tsv_path = os.path.join(derivatives_path,
                                   "labels/labels_covid_posi.tsv")
    labels_df = pd.read_csv(labels_tsv_path, sep="\t")

    # Convert the strings representing the list of labels to true lists
    labels_df["Labels"] = labels_df["Labels"].apply(get_labels_from_str)

    # Load DataFrame with informaton about the subjects
    subjects_tsv_path = os.path.join(args.sub_path, "participants.tsv")
    subjects_df = pd.read_csv(subjects_tsv_path, sep="\t")

    """
    Use the results from the tests (like PCR) to change the labels.

    Note: There are samples that are not labeled as "COVID 19" but the
          corresponding test for that subject for the same date of the
          scan (or a close day) is positive. So for some cases like
          "COVID 19 uncertain" or "pneumonia" is interesting to label
          those images as "COVID 19" (if the test is positive).
    """

    if not args.no_relabel:
        # Load the results of the COVID tests
        tests_path = os.path.join(args.sub_path,
                                  "derivatives/EHR/sil_reg_covid_posi.tsv")
        tests_df = pd.read_csv(tests_path, sep='\t')

        # Get only the tests that are positive
        posi_tests = tests_df[tests_df["result"] == "POSITIVO"].copy()
        # Convert dates from strings to datetime objects
        posi_tests["date"] = pd.to_datetime(posi_tests["date"],
                                            format="%d.%m.%Y")
        # Group tests by subject ID
        #  - Note: A subject can have several tests
        posi_tests_by_sub = posi_tests.groupby(["participant"])

        # Group the sessions labels by subject ID
        labels_by_sub = labels_df.groupby(["PatientID"])

        sess_labels_fixed = 0  # Counter to show stats

        # Iterate over the positive tests to check if we have to relabel
        print("Fixing labels with COVID tests results:")
        for sub_id, sub_tests in tqdm(posi_tests_by_sub):
            # Load subject sessions data
            sub_sessions_tsv = os.path.join(args.sub_path,
                                            sub_id,
                                            f"{sub_id}_sessions.tsv")
            # Check if the data exists
            if not os.path.isfile(sub_sessions_tsv):
                continue  # skip the subject

            # Load the sessions data of the current subject
            sub_sessions_df = pd.read_csv(sub_sessions_tsv, sep="\t")

            # Convert sessions dates from strings to datetime objects
            sub_sessions_df["study_date"] = pd.to_datetime(
                sub_sessions_df["study_date"], format="%Y%m%d")

            # Get the list of labels for each session of the subject
            #  Note: "ReportID" is the session ID
            sub_sessions_labels = labels_by_sub.get_group(sub_id)[['ReportID',
                                                                   'Labels']]

            # Compare the labels of each session with the COVID tests
            for idx, sess_row in sub_sessions_df.iterrows():
                sess_id = sess_row["session_id"]
                sess_date = sess_row["study_date"]

                # Get the list of labels of the current session
                sess_mask = sub_sessions_labels["ReportID"] == sess_id
                sess_labels_row = sub_sessions_labels[sess_mask]
                labels_list = sess_labels_row["Labels"].values[0]

                # Skip sessions with the COVID label (Nothing to change here)
                if 'COVID 19' in labels_list:
                    continue
                # Skip sessions with at least one of the labels to avoid
                if any(l in args.labels_to_avoid for l in labels_list):
                    continue
                # Skip sessions without at least one mandatory label
                if len(args.mandatory_labels):
                    # Look for a mandatory label
                    found = False
                    for l in labels_list:
                        if l in args.mandatory_labels:
                            found = True
                            break
                    # If not, skip the session
                    if not found:
                        continue

                # Look if any of the tests can affect the session labels
                for test_date in sub_tests["date"]:
                    # Compute time difference in days
                    days_diff = (sess_date - test_date).days

                    # Check if is a valid difference to fix the label
                    if -args.prev_days <= days_diff <= args.post_days:
                        # Add the COVID 19 label
                        #  Note: The extra [] are necessary
                        new_labels = [[labels_list + ["COVID 19"]]]
                        sessions_mask = labels_df["ReportID"] == sess_id
                        labels_df.loc[sessions_mask, "Labels"] = new_labels
                        sess_labels_fixed += 1
                        break  # Don't look for more tests

        print(f"Sessions with 'COVID 19' label added: {sess_labels_fixed}")

    else:
        print("Skipping the relabeling step")

    """
    Filter samples of interest to create the final Dataset.

        1 - The selected samples to create the ECVL Dataset must have at least
            one of the labels of interest (defined in args.target_labels).

        2 - The selected samples must be also anterior-posterior (AP) or
            posterior-anterior (PA) views.

        3 - If the clean-data argument is provided. We only take the images
            that are labeled as "OK".
    """

    # 1 - Filter by labels

    def filter_by_labels(df_row: pd.Series):
        """Returns True if at least one target label is in the row labels"""
        return any(label in df_row["Labels"] for label in args.target_labels)

    # Get the rows of the DataFrame that have at least one of the target labels
    samples_filter = labels_df.apply(filter_by_labels, axis=1)  # Rows mask
    selected_labels = labels_df[samples_filter].copy()

    # Prepare a function to extrat the label to classify each sample
    #   Note: In the labels_df dataframe each sample has a list of labels
    if args.multiclass:
        # Auxiliary set to compute the intersection with the samples labels
        target_labels_set = set(args.target_labels)

        def select_label(row_labels: pd.Series):
            # Get the labels from the sample that are in target_labels list
            row_labels_set = set(row_labels)
            return list(target_labels_set.intersection(row_labels_set))
    else:
        def select_label(row_labels: pd.Series):
            for label in args.target_labels:
                if label in row_labels:
                    return [label]  # Return the first match

    # Convert the lists of labels to a single label list
    selected_labels["Labels"] = selected_labels["Labels"].apply(select_label)

    # 2 - Filter by views (AP and PA)
    #   Note: We know the view by looking at the image file name

    def is_ap_or_pa(df_row: pd.Series):
        """If the sample row is an AP or PA image returns True, else False"""
        if "vp-ap" in df_row["filepath"] or "vp_pa" in df_row["filepath"]:
            return True
        return False

    # Get the rows of the DataFrame that are AP or PA images
    samples_filter = part_df.apply(is_ap_or_pa, axis=1)  # Rows mask filter
    selected_samples = part_df[samples_filter]

    # 3 - Get only the images that are manually validated

    if args.clean_data:
        # Load the dataframe with the annotations
        clean_df = pd.read_csv(args.clean_data, sep='\t')
        # Take only the samples that are "OK"
        ok_samples = clean_df[clean_df["status"] == "OK"]
        # Create the filter to take the "OK" samples only
        ok_filter = selected_samples["session"].isin(ok_samples["session"])
        # Apply the filter
        selected_samples = selected_samples[ok_filter]

    """
    Create the DataFrame with all the relevant info for each sample.
    The DataFrame contains:
        - Subject ID
        - Session ID
        - Path to the image
        - Labels list ("COVID 19", "normal", "pneumonia"...)
        - Patient gender
        - Patient age
    """

    main_df = pd.DataFrame(columns=['subject',
                                    'session',
                                    'filepath',
                                    'labels',
                                    'gender',
                                    'age'])

    # Prepare the folder to store the preprocessed images
    os.makedirs(os.path.join(args.sub_path, preproc_dirname), exist_ok=True)

    # We store the pairs of images paths (orig, dest) of the images that we are
    # going to preprocess. We do this to execute the preprocessing in parallel
    images_to_preprocess = []

    # Iterate over each sample to collect the data to create the new "main_df"
    print("Collecting samples data:")
    n_selected = len(selected_samples.index)
    for idx, row in tqdm(selected_samples.iterrows(), total=n_selected):
        sub_id = row["subject"]
        sess_id = row["session"]

        # Get the labels of the current sample
        row_labels = selected_labels[selected_labels["ReportID"] == sess_id]
        assert len(row_labels.index) <= 1  # Sanity check
        if len(row_labels.index) == 0:
            continue  # Skip the sample, we need the labels

        # Get the list of target labels ("COVID 19", "normal",...)
        row_labels = row_labels["Labels"].values[0]

        # This path must be relative to the folder of the output YAML
        new_img_relative_path = f"{preproc_dirname}/{sub_id}_{sess_id}_img.png"

        # Add the image to the preprocessing queue with the input and output
        # paths for the preprocessing function
        orig_img_path = os.path.join(args.sub_path, row['filepath'])
        new_img_path = os.path.join(args.sub_path, new_img_relative_path)

        if not os.path.isfile(new_img_path) or args.new_preproc:
            images_to_preprocess.append((orig_img_path, new_img_path))

        # Get subject data (age, gender...)
        sub_data = subjects_df[subjects_df["participant"] == sub_id]
        assert len(sub_data.index) == 1  # Sanity check
        sub_gender = sub_data["gender"].values[0]  # Get the str ('M' or 'F')
        sub_age = ast.literal_eval(sub_data["age"].values[0])[0]  # Get the int

        # Add the a row to the main DataFrame with the collected data
        new_row = {'subject': sub_id,
                   'session': sess_id,
                   'filepath': new_img_relative_path,
                   'labels': row_labels,
                   'gender': sub_gender,
                   'age': sub_age}
        main_df = main_df.append(new_row, ignore_index=True)

    # Apply the preprocessing to the images (in parallel)
    print("\nPreprocessing the images...")
    with mp.Pool(processes=args.n_proc) as pool:
        pool.starmap(histogram_equalization, images_to_preprocess, 10)
    print("Images preprocessed!")

    """
    Create the splits (training, validation, test).
        Note: The splits are made at subject level.
    """

    # Get a list with the subjects IDs shuffled (without repeated IDs)
    main_df_subjects = main_df["subject"].unique()
    np.random.shuffle(main_df_subjects)

    # Auxiliar values to compute the splits
    N = len(main_df_subjects)
    tr_end = args.splits[0]  # 0.6 (by default)
    val_end = tr_end + args.splits[1]  # 0.6 + 0.2 = 0.8 (by default)
    # The test split goes from 0.8 to 1.0 (by default)

    # Create the split train (60%), validation (20%), test (20%)
    train, val, test = np.split(main_df_subjects,
                                [int(tr_end * N), int(val_end * N)])

    # Create a new column in the main DataFrame to set the split of each sample
    main_df["split"] = ""

    splits = [("training", train), ("validation", val), ("test", test)]

    # Set the split values
    for name, split in splits:
        for sub_id in split:
            # Set the split value for all the samples of the current subject
            main_df.loc[main_df["subject"] == sub_id, "split"] = name

    print(f"\nTotal samples: {len(main_df.index)}")
    n_tr_samples = len(main_df[main_df['split'] == 'training'])
    print(f"Train split samples: {n_tr_samples}")
    n_val_samples = len(main_df[main_df['split'] == 'validation'])
    print(f"Validation split samples: {n_val_samples}")
    n_te_samples = len(main_df[main_df['split'] == 'test'])
    print(f"Test split samples: {n_te_samples}")

    # Store the new main DataFrame to a TSV
    main_df_outfile = os.path.join(args.sub_path, f"{args.yaml_name}.tsv")
    main_df.to_csv(main_df_outfile, sep='\t', index=False)
    print(f'\nStored splits data in "{main_df_outfile}"')

    """
    Prepare the YAML file to create the ECVL Dataset objects from the DataFrame
    created with all the informaton about the samples.
    """

    yaml_outfile = os.path.join(args.sub_path, f"{args.yaml_name}.yaml")
    stats_dict = create_ecvl_yaml(
        main_df, yaml_outfile, args.target_labels, args.multiclass)
    print(f'\nStored ECVL datset YAML in "{yaml_outfile}"')

    print("\nYAML labels count:")
    for label in stats_dict:
        print(f" - {label}: {stats_dict[label]}")


if __name__ == "__main__":
    # Script arguments handler
    arg_parser = argparse.ArgumentParser(
        description=("Prepares the YAML file to create the ECVL Dataset. "
                     "The YAML file will be placed in the sub-path provided"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        "--sub-path",
        help="Path to the directory with the subjects data folders",
        default="../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/")

    arg_parser.add_argument(
        "--clean-data",
        help=("Path to the TSV file with the annotations generated wit the "
              "data_cleaning.ipynb notebook"),
        default=None)

    arg_parser.add_argument(
        "--no-relabel",
        help=("By default some samples are relabeled using the COVID tests "
              "that are positive and with a close date to the scan day. The "
              "days margin can be configured with the flags: "
              "--prev-days and --post-days"),
        action="store_true")

    arg_parser.add_argument(
        "--prev-days",
        help=("Number of previous days before the scan day to take the COVID "
              "test result as relevant. '0' means the same day."),
        default=0,
        type=int)

    arg_parser.add_argument(
        "--post-days",
        help=("Number of days after the scan day to take the COVID "
              "test result as relevant. '0' means the same day."),
        default=0,
        type=int)

    arg_parser.add_argument(
        "--mandatory-labels",
        help=("List of labels that must be present in a sample to be able to "
              "relabel it using the COVID tests results"),
        metavar="label_name",
        nargs='*',
        default=["COVID 19 uncertain", "pneumonia", "infiltrates"],
        type=str)

    arg_parser.add_argument(
        "--labels-to-avoid",
        help=("List of labels that must NOT be present in a sample to be able "
              "to relabel it using the COVID tests results"),
        metavar="label_name",
        nargs='*',
        default=["exclude", "normal"],
        type=str)

    arg_parser.add_argument(
        "--seed",
        help="Seed value to shuffle the dataset",
        default=1234,
        type=int)

    arg_parser.add_argument(
        "--target-labels",
        help=("Target labels to select the samples for training. "
              "The order is important, the first matching label will be taken "
              "as the label for the sample."),
        metavar="label_name",
        nargs='+',
        default=["normal", "COVID 19", "pneumonia", "infiltrates"],
        type=str)

    arg_parser.add_argument(
        "--multiclass",
        help=("To create multiclass labels instead of taking one label from "
              "each sample"),
        action="store_true")

    arg_parser.add_argument(
        "--splits",
        help="Size of the train, validation and test splits",
        metavar=("train_size", "val_size", "test_size"),
        nargs=3,
        default=[0.6, 0.2, 0.2],
        type=float)

    arg_parser.add_argument(
        "--n-proc",
        help="Number of processes to use for preprocessing the data",
        default=mp.cpu_count(),
        type=int)

    arg_parser.add_argument(
        "--new-preproc",
        help=("If not set the script tries to use a previously preprocessed "
              "image (if it exists), else the preprocessing is always done"),
        action="store_true")

    arg_parser.add_argument(
        "--yaml-name",
        help=("Name of the YAML file to create (without the file extension). "
              "This name is also used to create the folder for the "
              "preprocessed data ('{YAML_NAME}_preproc_data')"),
        default="ecvl_bimcv_covid19",
        type=str)

    main(arg_parser.parse_args())
