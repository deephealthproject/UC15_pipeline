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


def create_ecvl_yaml(df: pd.DataFrame, yaml_path: str, target_labels: list):
    """
    Given a DataFrame with all the necessary data to create a training dataset,
    this function creates a YAML file to be able to create the corresponding
    ECVL Dataset object.

    Args:
        df: A DataFrame object with the columns: "subject", "session",
            "filepath", "labels", "gender", "age" and "split".

        yaml_path: Path to store the created YAML file.

        target_labels: A list with the labels to use for classification. The
                       labels that are not present in this list will be avoided
                       when creating the YAML samples.
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

    for sample_idx, (_, row) in enumerate(df.iterrows()):
        # Get only the labels that belong to the target_labels list provided
        labels = list(target_labels_set.intersection(set(row["labels"])))
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
