"""
Module with auxiliary functions to process the data that does not come
from images.
"""
import ast
import yaml

import pandas as pd


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

    # Auxiliary set object to compute intersections of labels sets
    target_labels_set = set(target_labels)

    counts_dict = {}  # Counts by label to return at the end

    for sample_idx, (_, row) in enumerate(df.iterrows()):
        # Get the labels for the sample
        if multiclass:
            labels = list(target_labels_set.intersection(set(row["labels"])))
            label_name = "_and_".join(sorted(labels))
            counts_dict[label_name] = counts_dict.get(label_name, 0) + 1
        else:
            labels = []
            for label in target_labels:
                if label in row["labels"]:
                    labels.append(label)
                    counts_dict[label] = counts_dict.get(label, 0) + 1
                    break  # We take the first matching label

        assert len(labels) > 0  # Sanity check

        # Store the sample data
        samples.append({"location": row["filepath"],
                        "label": labels,
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
