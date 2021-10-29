"""
Module with auxiliary functions to process the data that does not come
from images.
"""
import os
import ast
import yaml

import pandas as pd
from tqdm import tqdm


def get_labels_from_str(labels_str: str, verbose: bool = False) -> list:
    """
    Given the string representation of a list of strings returns a list
    object with the string labels. The labels are trimed to remove
    undesired spaces.

    Args:
        labels_str: A string representation of a list of strings.

        verbose: To enable or disable error messages.

    Returns:
        The list represented by "labels_str" with the strings in it trimed.
    """
    try:
        # From string to list
        labels_list = ast.literal_eval(labels_str)
    except:
        # Some labels are not complete and fail when parsing to list
        if verbose:
            print(f'Failed to process labels: "{labels_str}"')
        return []

    # Remove extra spaces from labels
    return list(map(str.strip, labels_list))


def relabel_with_covid_tests(posi_labels_df: pd.DataFrame, args) -> int:
    """
    Relabels the sessions that are not labeled as 'COVID 19' if they met the
    conditions to be relabeled using the covid tests (like PCR) results.

    Note: Only the 'posi' patients have COVID tests.

    Args:
        posi_labels_df: DataFrame with the labels of the sessions that are going
                        to be relabeled.

        args: The 'args' variable created in the main preprocessing script that
              contains the rest of the necessary informaton.

    Returns:
        The number of sessions that got relabeled.
    """
    # Load the results of the COVID tests
    tests_path = os.path.join(args.posi_path,
                              "derivatives/EHR/sil_reg_covid_posi.tsv")
    tests_df = pd.read_csv(tests_path, sep='\t')

    # Get only the tests that are positive
    positive_tests = tests_df[tests_df["result"] == "POSITIVO"].copy()
    # Convert dates from strings to datetime objects
    positive_tests["date"] = pd.to_datetime(positive_tests["date"],
                                            format="%d.%m.%Y")
    # Group tests by subject ID
    #  - Note: A subject can have several tests
    positive_tests_by_sub = positive_tests.groupby(["participant"])

    # Group the sessions labels by subject ID
    labels_by_sub = posi_labels_df.groupby(["PatientID"])

    sess_labels_fixed = 0  # Counter to show stats

    # Iterate over the positive tests to check if we have to relabel
    print("Fixing labels with COVID tests results:")
    for sub_id, sub_tests in tqdm(positive_tests_by_sub):
        # Load subject sessions data
        sub_sessions_tsv = os.path.join(args.posi_path,
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
                    # Prepare the mask to select the session images
                    sessions_mask = posi_labels_df["ReportID"] == sess_id
                    # Add the COVID 19 label
                    posi_labels_df.loc[sessions_mask, "Labels"] = posi_labels_df[sessions_mask]["Labels"].apply(lambda l: l + ["COVID 19"])
                    # Update the counter of samples changed
                    sess_labels_fixed += sessions_mask.sum()
                    break  # Don't look for more tests

    return sess_labels_fixed


def create_ecvl_yaml(df: pd.DataFrame,
                     yaml_path: str,
                     target_labels: list,
                     multiclass: bool) -> dict:
    """
    Given a DataFrame with all the necessary data to create a training dataset,
    this function creates a YAML file to be able to create the corresponding
    ECVL Dataset object.

    Args:
        df: A DataFrame object with the columns: "subject", "session",
            "filepath", "labels", "gender", "age" and "split".

        yaml_path: Path to store the created YAML file.

        target_labels: A list with the labels to use for classification. The
                       order is important, the first matching label will be
                       taken as the label for the sample.

        multiclass: If True, instead of taking just the first matching label
                    for each sample, all the matching labels will be taken.

    Returns:
        A dictionary with the counts of the samples by label.
    """

    yaml_outstream = open(yaml_path, 'w')  # YAML output file stream

    # To avoid undesired tokens (like *id001 or &id001) generated by yaml
    yaml.Dumper.ignore_aliases = lambda *args: True

    # Write the dataset name
    yaml.dump({"name": "BIMCV COVID 19+"}, yaml_outstream)

    # Write the classes names (of the training labels)
    yaml.dump({"classes": target_labels},
              yaml_outstream,
              default_flow_style=None)

    # Write the aditional features names
    yaml.dump({"features": ["age", "gender"]},
              yaml_outstream,
              default_flow_style=None)

    samples = []  # To store the samples data
    # To store the index of each sample in their corresponding split
    splits_indexes = {split_name: list()
                      for split_name in df["split"].unique()}

    counts_dict = {}  # Counts by label to return at the end

    for sample_idx, (_, row) in enumerate(df.iterrows()):
        # Update the classes counter
        if multiclass:
            label_name = " + ".join(sorted(row["labels"]))
        else:
            label_name = row["labels"][0]
        counts_dict[label_name] = counts_dict.get(label_name, 0) + 1

        # Store the sample data
        samples.append({"location": row["filepath"],
                        "label": row["labels"],
                        "values": {"age": row["age"],
                                   "gender": row["gender"]}})

        # Store the sample ID in its corresponding split
        splits_indexes[row["split"]].append(sample_idx)

    # Write the samples data
    yaml.dump({"images": samples},
              yaml_outstream,
              default_flow_style=None)

    # Write the splits indexes
    yaml.dump({"split": splits_indexes},
              yaml_outstream,
              default_flow_style=None)

    yaml_outstream.close()

    return counts_dict
