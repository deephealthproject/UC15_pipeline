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
import random
import json
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.image_processing import preprocess_image, create_copy_with_DA
from lib.data_processing import get_labels_from_str, create_ecvl_yaml, relabel_with_covid_tests


def main(args):
    # Set the seeds to be able to replicate the results
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Check arguments
    assert sum(args.splits) == 1, "The splits values sum must be 1.0!"

    """
    Load and prepare data
    """
    posi_derivatives_path = os.path.join(args.posi_path, "derivatives")
    neg_derivatives_path = os.path.join(args.neg_path, "derivatives")

    # Load DataFrames with all the images (paths) by session and subject
    cols = ["subject", "session", "filepath"]  # Fix the original columns
    posi_df = pd.read_csv(os.path.join(posi_derivatives_path, "partitions.tsv"),
                          sep="\t",
                          header=0,
                          names=cols)
    neg_df = pd.read_csv(os.path.join(neg_derivatives_path, "partitions.tsv"),
                         sep="\t",
                         header=0,
                         names=cols)

    # Load DataFrames with the labels for each session
    posi_labels_path = os.path.join(posi_derivatives_path, "labels/labels_covid_posi.tsv")
    neg_labels_path = os.path.join(neg_derivatives_path, "labels/Labels_covid_NEG_JAN21.tsv")
    posi_labels_df = pd.read_csv(posi_labels_path, sep="\t")
    neg_labels_df = pd.read_csv(neg_labels_path, sep="\t")

    # Convert the strings representing the list of labels to true lists
    posi_labels_df["Labels"] = posi_labels_df["Labels"].apply(get_labels_from_str)
    neg_labels_df["Labels"] = neg_labels_df["Labels"].apply(get_labels_from_str)

    # Load DataFrames with informaton about the subjects
    posi_sub_df = pd.read_csv(os.path.join(args.posi_path, "participants.tsv"),
                              sep="\t")
    neg_sub_df = pd.read_csv(os.path.join(args.neg_path, "participants.tsv"),
                             sep="\t")

    """
    Use the results from the tests (like PCR) to change the labels.

    ** Only available for 'posi' patients **

    Note: There are samples that are not labeled as "COVID 19" but the
          corresponding test for that subject for the same date of the
          scan (or a close day) is positive. So for some cases like
          "COVID 19 uncertain" or "pneumonia" is interesting to label
          those images as "COVID 19" (if the test is positive).
    """

    if not args.no_relabel:
        sess_labels_fixed = relabel_with_covid_tests(posi_labels_df, args)
        print(f"Sessions with 'COVID 19' label added: {sess_labels_fixed}")
    else:
        print("Skipping the relabeling step")

    """
    Filter samples of interest to create the final Dataset.

        1 - The selected samples to create the ECVL Dataset must have at least
            one of the labels of interest. The labels in args.posi_target_labels
            apply to the patients in the 'posi' dataset while the labels in
            args.neg_target_labels apply to the patients in the 'neg' dataset.

        2 - The selected samples must be also anterior-posterior (AP) or
            posterior-anterior (PA) views

        3 - If the clean-data argument is provided. We only take the images
            that are labeled as "OK" in the TSV provided by the clean-data flag
            Note: Only for posi patients

        4 - If the only-dx or only-cr (exclusive or) flag is enabled we have to
            select only the corresponding samples

        5 - Remove the samples that are in the list of images to remove (
            provided with the flag --images-to-avoid)
    """

    # 1 - Filter by labels

    # Filter posi patients
    def filter_by_labels(df_row: pd.Series):
        """Returns True if at least one target label is in the row labels"""
        return any(label in df_row["Labels"] for label in args.posi_target_labels)

    # Get the rows of the DataFrame that have at least one of the target labels
    samples_filter = posi_labels_df.apply(filter_by_labels, axis=1)  # Rows mask
    posi_selected_labels = posi_labels_df[samples_filter].copy()

    # Filter neg patients
    def filter_by_labels(df_row: pd.Series):
        """Returns True if at least one target label is in the row labels"""
        return any(label in df_row["Labels"] for label in args.neg_target_labels)

    # Get the rows of the DataFrame that have at least one of the target labels
    samples_filter = neg_labels_df.apply(filter_by_labels, axis=1)  # Rows mask
    neg_selected_labels = neg_labels_df[samples_filter].copy()

    # Prepare a function to extract the label or labels to classify each sample
    if args.multiclass:
        # Select the list of labels that represents each session
        def select_label(row_labels: pd.Series, target_labels: list):
            # Auxiliary sets to compute the intersection with the samples labels
            target_labels_set = set(target_labels)
            row_labels_set = set(row_labels)

            # Get the labels from the sample that are in target_labels list
            selected_lab = list(target_labels_set.intersection(row_labels_set))
            if len(selected_lab) == 0:
                raise Exception("Unexpected error. No target label found!")
            return sorted(selected_lab)
    else:
        # Select the label that represents each session
        def select_label(row_labels: pd.Series, target_labels: list):
            if len(row_labels) == 1 and row_labels[0] == "":
                return ["normal"]

            for label in target_labels:
                if label in row_labels:
                    return [label]  # Return the first match

            raise Exception("Unexpected error. No target label found!")

    # Convert the lists of labels to a list with only the target label
    posi_selected_labels["Labels"] = posi_selected_labels["Labels"].apply(
        lambda x: select_label(x, args.posi_target_labels))
    neg_selected_labels["Labels"] = neg_selected_labels["Labels"].apply(
        lambda x: select_label(x, args.neg_target_labels))

    # 2 - Filter by views (AP and PA)
    #   Note: We know the view by looking at the image file name

    def is_ap_or_pa(df_row: pd.Series):
        """If the sample row is an AP or PA image returns True, else False"""
        if "vp-ap" in df_row["filepath"] or "vp_pa" in df_row["filepath"]:
            return True
        return False

    # Get the rows of the DataFrame that are AP or PA images
    samples_filter = posi_df.apply(is_ap_or_pa, axis=1)  # Rows mask filter
    posi_selected_samples = posi_df[samples_filter]
    samples_filter = neg_df.apply(is_ap_or_pa, axis=1)  # Rows mask filter
    neg_selected_samples = neg_df[samples_filter]

    # 3 - Get only the images that are manually validated

    if args.clean_data:
        # Load the dataframe with the annotations
        clean_df = pd.read_csv(args.clean_data, sep='\t')
        # Take only the samples that are "OK"
        ok_samples = clean_df[clean_df["status"] == "OK"]
        # Create the filter to take the "OK" samples only
        ok_filter = posi_selected_samples["session"].isin(ok_samples["session"])
        # Apply the filter
        posi_selected_samples = posi_selected_samples[ok_filter]

    # 4 - Get only the DX or CR images

    if args.only_dx != args.only_cr:
        # Select the target image type
        type_ = "dx" if args.only_dx else "cr"

        def aux_func(row):
            return type_ in row["filepath"]
        # Get the rows of the DataFrame that are from the selected type
        samples_filter = posi_selected_samples.apply(aux_func, axis=1)
        posi_selected_samples = posi_selected_samples[samples_filter]
        samples_filter = neg_selected_samples.apply(aux_func, axis=1)
        neg_selected_samples = neg_selected_samples[samples_filter]
    elif args.only_dx:  # Both are True
        raise Exception(
            "You can only enable one of the flags '--only-dx' or '--only-cr'")

    # 5 - Remove the images to avoid (argument --images-to-avoid)

    if args.images_to_avoid:
        print("\nGoing to remove the images to avoid listed "
              f"in '{args.images_to_avoid}'")
        # Use regular expresions to get the session ID of the sample
        # Note: The sessions IDs are unique and can be used to filter
        ses_regex = re.compile(r'ses-E\d+')
        with open(args.images_to_avoid, 'r') as images_list:
            sessions2avoid = []  # To store the invalid sessions
            for line in images_list.readlines():
                ses_id = ses_regex.findall(line)
                assert len(ses_id)  # There must be at least one match always
                sessions2avoid.append(ses_id[0])

        # Filter the samples dataframe
        # posi patients
        n_before = len(posi_selected_samples.index)
        rows2avoid_filter = posi_selected_samples['session'].isin(sessions2avoid)
        posi_selected_samples = posi_selected_samples[~rows2avoid_filter]
        n_after = len(posi_selected_samples.index)
        n_removed = n_before - n_after
        # neg patients
        n_before = len(neg_selected_samples.index)
        rows2avoid_filter = neg_selected_samples['session'].isin(sessions2avoid)
        neg_selected_samples = neg_selected_samples[~rows2avoid_filter]
        n_after = len(neg_selected_samples.index)
        n_removed += n_before - n_after
        print(f"Removed {n_removed} samples")

    """
    Merge the 'posi' and 'neg' dataframes now that we have done some specific
    filtering on them so the next steps of the script are clearer.

    We are also going to add some extra columns to simplify the processing
    in the following steps of the script:

        - full_path: Full path to access the images in order be able to
                     preprocess them. We will add the path to the 'posi' or
                     'neg' folders depending on the patient type of the image.

        - orig_subject: The 'posi' and 'neg' sets of patients have some common
                        patients but the subject ID is not the same in both sets.
                        The dataset provides a file with the subjects IDs
                        translation between sets. In this 'orig_subject' column
                        we will store the original ID, and in the 'subject' column
                        we will apply the translation following the file provided.

        - patient_type: To mark each sample as 'posi' or 'neg' before the merge.
    """
    # Copy the full dataframes to avoid warnings from pandas for modifying a copy of a slice
    posi_selected_samples = posi_selected_samples.copy()
    neg_selected_samples = neg_selected_samples.copy()

    # 1. Merge the selected samples

    # 1.1 Add the column with the full paths to the posi samples dataframe
    def posi_to_full_path(path: str) -> str:
        return os.path.join(args.posi_path, path)
    posi_selected_samples["full_path"] = posi_selected_samples['filepath'].apply(posi_to_full_path)

    # 1.2 Add the column with the full paths to the neg samples dataframe
    def neg_to_full_path(path: str) -> str:
        full_path = os.path.join(args.neg_path, path)
        # Add a "mod-rx" folder before the file name to match the real
        # structure of the dataset
        parts = full_path.split('/')
        return '/'.join([*parts[:-1], "mod-rx", parts[-1]])
    neg_selected_samples["full_path"] = neg_selected_samples['filepath'].apply(neg_to_full_path)

    # 1.3 Add the columns with the original subject ID for each sample
    posi_selected_samples["orig_subject"] = posi_selected_samples['subject']
    neg_selected_samples["orig_subject"] = neg_selected_samples['subject']

    # 1.4 Change the common subjects IDs in the 'neg' dataframe with the
    #     corresponding ID in the 'posi' dataset
    common_subs_df = pd.read_csv(args.common_ids, sep='\t')
    neg_comm_subs = common_subs_df['covid19_neg'].values
    posi_comm_subs = common_subs_df['covid19_posi'].values
    neg_selected_samples['subject'] = neg_selected_samples['subject'].replace(neg_comm_subs, posi_comm_subs)

    # 1.5 Add the 'patient_type' column
    posi_selected_samples['patient_type'] = 'posi'
    neg_selected_samples['patient_type'] = 'neg'

    # 1.6 Concat the 'posi' and 'neg' dataframes
    selected_samples = pd.concat([posi_selected_samples, neg_selected_samples],
                                 ignore_index=True)

    # Merge the selected labels
    # posi_selected_labels, neg_selected_labels
    selected_labels = pd.concat([posi_selected_labels, neg_selected_labels],
                                 ignore_index=True)

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
    preproc_dirname = f"{args.yaml_name}_preproc_data"
    os.makedirs(os.path.join(args.preproc_path, preproc_dirname), exist_ok=True)

    # We store the pairs of images paths (orig, dest) of the images that we are
    # going to preprocess. We do this to execute the preprocessing in parallel
    images_to_preprocess = []

    if not args.avoid_preproc:
        run_regex = re.compile(r'run-\d+')
        acq_regex = re.compile(r'acq-\d+')

    # Iterate over each sample to collect the data to create the new "main_df"
    print("\nCollecting samples data:")
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

        if not args.avoid_preproc:
            # Get the run and acq number
            run_number = run_regex.findall(row['filepath'])
            acq_number = acq_regex.findall(row['filepath'])
            assert len(run_number) == 1  # Only one match should be present
            assert len(acq_number) <= 1   # The acq number may not be present

            # This path must be relative to the folder of the output YAML
            if len(acq_number) > 0:
                new_img_relpath = f"{preproc_dirname}/{sub_id}_{sess_id}_{run_number[0]}_{acq_number[0]}.png"
            else:
                new_img_relpath = f"{preproc_dirname}/{sub_id}_{sess_id}_{run_number[0]}.png"

            # Add the image to the preprocessing queue with the input and output
            # paths for the preprocessing function
            new_img_path = os.path.join(args.preproc_path, new_img_relpath)

            # Add the image to the preprocessing queue
            if not os.path.isfile(new_img_path) or args.new_preproc:
                # See preprocess_image() args
                extra_args = ("adaptive", True, args.to_rgb, args.colormap, args.n_colors)
                images_to_preprocess.append((row['full_path'],
                                             new_img_path,
                                             *extra_args))
        else:
            new_img_relpath = row['filepath']  # Use the original image

        # Get subject data (age, gender...)
        if row['patient_type'] == 'posi':
            sub_data = posi_sub_df[posi_sub_df["participant"] == row['orig_subject']]
        else:  # is 'neg'
            sub_data = neg_sub_df[neg_sub_df["participant"] == row['orig_subject']]
        assert len(sub_data.index) == 1  # Sanity check
        sub_gender = sub_data["gender"].values[0]  # Get the str ('M' or 'F')
        sub_age = ast.literal_eval(sub_data["age"].values[0])[0]  # Get the int

        # Add the a row to the main DataFrame with the collected data
        new_row = {'subject': sub_id,
                   'session': sess_id,
                   'filepath': new_img_relpath,
                   'labels': row_labels,
                   'gender': sub_gender,
                   'age': sub_age}
        main_df = main_df.append(new_row, ignore_index=True)

    # Apply the preprocessing to the images (in parallel)
    #  - The preprocessing steps applied are:
    #      1. Invert the image pixels if the original image is inverted. We do
    #         this by looking at the image metadata.
    #
    #      2. Normalize the image using histogram equalization
    if not args.avoid_preproc:
        print("\nPreprocessing the images...")
        with mp.Pool(processes=args.n_proc) as pool:
            pool.starmap(preprocess_image, images_to_preprocess, 10)
        print("Images preprocessed!")
    else:
        assert len(images_to_preprocess) == 0  # Sanity check
        print("\nAvoiding image preprocessing. Going to use the paths to the "
              "original images")

    """
    Create the splits (training, validation, test).
        Note: The splits are made at subject level.
    """
    # Combine all the labels in one list
    all_target_labels = list(set(args.posi_target_labels + args.neg_target_labels))

    if not args.avoid_balancing:
        print("\nGoing to create the balanced validation and test partitions")
        """
        Prepare the data structure to balance the data splits
        """
        data_by_sub = main_df.groupby("subject")
        label2idx = {label: i for i, label in enumerate(all_target_labels)}
        # To collect the data for balancing the splits. A list per target label
        subs_labels = [[] for _ in range(len(all_target_labels))]

        for sub_id, sub_sessions in data_by_sub:
            # Count the number of sessions for each label
            sub_labels_counter = np.array([0] * len(all_target_labels))
            for idx, sess in sub_sessions.iterrows():
                for label in sess["labels"]:
                    sub_labels_counter[label2idx[label]] += 1

            # This is the label that will be taken into account when filling
            # the patitions in a balanced way
            main_label = np.argmax(sub_labels_counter)
            # Save the subject data
            subs_labels[main_label].append((sub_id, sub_labels_counter))

        # Sort each list by number of sessions for each subject
        for subs_l in subs_labels:
            subs_l.sort(key=lambda x: sum(x[1]))

        """
        Define the splits by creating balanced validation and tests splits, and
        putting the rest of the data into the training split
        """
        # Define the size of the partitions
        n_total_samples = len(main_df.index)
        n_val = int(n_total_samples * args.splits[1])
        n_test = int(n_total_samples * args.splits[2])

        # Fill the validation split
        val_subjects = []
        n_val_samples = 0
        val_labels = np.array([0] * len(all_target_labels))
        min_label = 0
        while n_val_samples < n_val:
            # Take a subject to add samples to the current minimum label
            try:
                sub_id, labels = subs_labels[min_label].pop()
            except IndexError as err:
                label_name = all_target_labels[min_label]
                my_err = Exception(("The are not enought samples of the "
                                    f"class {label_name} to create a "
                                    " balanced validation split"))
                raise my_err from err
            # Add the subject to the validation split
            val_subjects.append(sub_id)
            # Update the number of samples for each label
            val_labels += labels
            # Update the new total number of samples
            n_val_samples = sum(val_labels)
            # Take the new minimum label
            min_label = np.argmin(val_labels)

        # Fill the test split
        test_subjects = []
        n_test_samples = 0
        test_labels = np.array([0] * len(all_target_labels))
        min_label = 0
        while n_test_samples < n_test:
            # Take a subject to add samples to the current minimum label
            try:
                sub_id, labels = subs_labels[min_label].pop()
            except IndexError as err:
                label_name = all_target_labels[min_label]
                my_err = Exception(("The are not enought samples of the "
                                    f"class {label_name} to create a "
                                    "balanced test split"))
                raise my_err from err

            # Add the subject to the test split
            test_subjects.append(sub_id)
            # Update the number of samples for each label
            test_labels += labels
            # Update the new total number of samples
            n_test_samples = sum(test_labels)
            # Take the new minimum label
            min_label = np.argmin(test_labels)

        # Put the rest of the samples into the train split
        train_subjects = []
        train_labels = np.array([0] * len(all_target_labels))
        for subs_l in subs_labels:
            for sub_id, labels in subs_l:
                # Add the subject to the training split
                train_subjects.append(sub_id)
                # Update the number of samples for each label
                train_labels += labels

        n_train_samples = sum(train_labels)

        print("\nTrain split:")
        print(f"  + number of samples: {n_train_samples}")
        print("  + labels distribution:")
        for i, count in enumerate(train_labels):
            print(f"    - {all_target_labels[i]}: {count}")
        print(f"  + number of subjects: {len(train_subjects)}")

        print("\nValidation split:")
        print(f"  + number of samples: {n_val_samples}")
        print("  + labels distribution:")
        for i, count in enumerate(val_labels):
            print(f"    - {all_target_labels[i]}: {count}")
        print(f"  + number of subjects: {len(val_subjects)}")

        print("\nTest split:")
        print(f"  + number of samples: {n_test_samples}")
        print("  + labels distribution:")
        for i, count in enumerate(test_labels):
            print(f"    - {all_target_labels[i]}: {count}")
        print(f"  + number of subjects: {len(test_subjects)}")

    else:  # args.avoid_balancing is True
        print("\nGoing to create the partitions randomly (without balancing)")
        # Get a list with the subjects IDs shuffled (without repeated IDs)
        main_df_subjects = main_df["subject"].unique()
        np.random.shuffle(main_df_subjects)

        # Auxiliar values to compute the splits
        N = len(main_df_subjects)
        tr_end = args.splits[0]  # 0.7 (by default)
        val_end = tr_end + args.splits[1]  # 0.7 + 0.15 = 0.85 (by default)
        # The test split goes from 0.85 to 1.0 (by default)

        # Create the split train (70%), validation (15%), test (15%)
        sub_splits = np.split(main_df_subjects,
                              [int(tr_end * N), int(val_end * N)])
        train_subjects, val_subjects, test_subjects = sub_splits

    # Create a new column in the main DataFrame to set the split of each sample
    main_df["split"] = ""

    splits = [("training", train_subjects),
              ("validation", val_subjects),
              ("test", test_subjects)]

    # Set the split values
    for name, split in splits:
        for sub_id in split:
            # Set the split value for all the samples of the current subject
            main_df.loc[main_df["subject"] == sub_id, "split"] = name

    """
    Balance the training split.

    The strategies available are:
        - oversampling: The samples of the minority classes are replicated
                        randomly using data augmentaion until the number of
                        samples in each class is equal to the majority class

        - undersampling: The samples of the majority class (or classes) are
                         randomly deleted until reaching the same number of
                         samples than the minority class
    """

    if args.train_balancing_type is not None and not args.avoid_balancing:
        if args.train_balancing_type == "oversampling":
            print("\nGoing to user oversampling to balance the train split:")
            # We store the pairs of images paths (orig, dest) of the images
            # that we are going to copy applying data augmentaion.
            images_to_aug = []

            # Select the samples of the training split
            train_samples = main_df[main_df["split"] == "training"]

            # Get the index and number of samples of the majority label
            max_label = np.argmax(train_labels)
            n_max_label = train_labels[max_label]
            print((f" - The majority class is {all_target_labels[max_label]} "
                   f"with {n_max_label} samples"))
            for label_idx, label in enumerate(all_target_labels):
                if label_idx == max_label:
                    continue  # We dont't have to replicate any sample

                # Get the samples of the current label in the training split
                samples_filter = train_samples.apply(
                    lambda row: label in row["labels"], axis=1)
                label_samples = train_samples[samples_filter]

                # Select random samples to copy
                n_samples = len(label_samples.index)
                n_to_copy = n_max_label - n_samples
                assert n_to_copy >= 0  # Sanity check
                print((f" - Class {label} has {n_samples}, "
                       f"going to create {n_to_copy} to reach {n_max_label}"))

                # To see if we have to create extra copies for some samples
                extra_copies = False
                # Compute the fixed number of copies that we have to make
                # for each sample of the minority label
                if n_to_copy < n_samples:
                    n_copies = 1
                    rand_select = random.sample(range(n_samples), n_to_copy)
                    samples_to_copy = label_samples.iloc[rand_select]
                elif n_to_copy == n_samples:
                    n_copies = 1
                    samples_to_copy = label_samples
                else:
                    n_copies = (n_to_copy // n_samples)
                    samples_to_copy = label_samples
                    extra_copies = True

                # Add the fixed number of copies for each sample
                for idx, row in samples_to_copy.iterrows():
                    orig_path = row["filepath"]
                    for copy_idx in range(n_copies):
                        if not args.avoid_preproc:
                            # Create the filepath of the new copy
                            # Note: This path is relative to the YAML file
                            new_path = orig_path[:-4] + f"_DA-{copy_idx}.png"

                            # Create the paths for creating the copies
                            orig_img_path = os.path.join(args.preproc_path,
                                                         orig_path)
                            new_img_path = os.path.join(args.preproc_path,
                                                        new_path)

                            # Add the sample to the data augmentaion queue
                            if not os.path.isfile(new_img_path) or args.new_preproc:
                                images_to_aug.append((orig_img_path,
                                                      new_img_path))
                        else:
                            new_path = orig_path  # Use the original image

                        # Add the copy to the main dataframe
                        new_row = {'subject': row["subject"],
                                   'session': row["session"],
                                   'filepath': new_path,
                                   'labels': row["labels"],
                                   'gender': row["gender"],
                                   'age': row["age"],
                                   'split': row["split"]}
                        main_df = main_df.append(new_row, ignore_index=True)

                if extra_copies:
                    # Compute the number of missing copies
                    n_extra = n_to_copy - (n_copies * n_samples)
                    assert 0 < n_extra < n_samples  # Sanity check
                    # Select randomly the samples to make the copies
                    rand_select = random.sample(range(n_samples), n_extra)
                    samples_to_copy = label_samples.iloc[rand_select]
                    copy_idx = n_copies
                    for idx, row in samples_to_copy.iterrows():
                        orig_path = row["filepath"]
                        if not args.avoid_preproc:
                            # Create the filepath of the new copy
                            # Note: This path is relative to the YAML file
                            new_path = orig_path[:-4] + f"_DA-{copy_idx}.png"

                            # Create the paths for creating the copies
                            orig_img_path = os.path.join(args.preproc_path,
                                                         orig_path)
                            new_img_path = os.path.join(args.preproc_path,
                                                        new_path)

                            # Add the sample to the data augmentaion queue
                            if not os.path.isfile(new_img_path) or args.new_preproc:
                                images_to_aug.append((orig_img_path,
                                                      new_img_path))
                        else:
                            new_path = orig_path  # Use the original image

                        # Add the copy to the main dataframe
                        new_row = {'subject': row["subject"],
                                   'session': row["session"],
                                   'filepath': new_path,
                                   'labels': row["labels"],
                                   'gender': row["gender"],
                                   'age': row["age"],
                                   'split': row["split"]}
                        main_df = main_df.append(new_row, ignore_index=True)

            # Apply the augmentaions to the images (in parallel)
            if not args.avoid_preproc:
                print("\nCreating augmented copies of the images...")
                with mp.Pool(processes=args.n_proc) as pool:
                    pool.starmap(create_copy_with_DA, images_to_aug, 10)
                print("Images created!")
            else:
                assert len(images_to_aug) == 0  # Sanity check
                print("\nAvoiding the creation of copies with DA to balance "
                      "the dataset using oversampling. Going to create "
                      "regular copies instead")

        elif args.train_balancing_type == "undersampling":
            raise Exception("Undersampling is not implemented!")
        else:
            raise Exception(("The balancing strategy provided for the train "
                             "split is not valid"))

    # Show a summary of the number of samples by split
    print(f"\nTotal samples: {len(main_df.index)}")
    n_tr_samples = len(main_df[main_df['split'] == 'training'])
    print(f"Train split samples: {n_tr_samples}")
    n_val_samples = len(main_df[main_df['split'] == 'validation'])
    print(f"Validation split samples: {n_val_samples}")
    n_te_samples = len(main_df[main_df['split'] == 'test'])
    print(f"Test split samples: {n_te_samples}")

    # Store the new main DataFrame to a TSV
    main_df_outfile = os.path.join(args.preproc_path, f"{args.yaml_name}.tsv")
    main_df.to_csv(main_df_outfile, sep='\t', index=False)
    print(f'\nStored splits data in "{main_df_outfile}"')

    """
    Prepare the YAML file to create the ECVL Dataset objects from the DataFrame
    created with all the informaton about the samples.
    """

    yaml_outfile = os.path.join(args.preproc_path, f"{args.yaml_name}.yaml")
    stats_dict = create_ecvl_yaml(main_df, yaml_outfile, all_target_labels, args.multiclass)
    print(f'\nStored ECVL datset YAML in "{yaml_outfile}"')

    # Store the script config
    config_outfile = os.path.join(args.preproc_path, f"{args.yaml_name}_args.json")
    with open(config_outfile, 'w') as fstream:
        json.dump(vars(args), fstream, indent=4, sort_keys=True)
    print(f'\nStored the dataset config in "{config_outfile}"')

    print("\nYAML total labels count:")
    for label in stats_dict:
        print(f" - {label}: {stats_dict[label]}")


if __name__ == "__main__":
    # Script arguments handler
    arg_parser = argparse.ArgumentParser(
        description=("Prepares the YAML file to create the ECVL Dataset. "
                     "The YAML file will be placed in the sub-path provided"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        "--posi-path",
        help="Path to the directory with the data of the positive patients",
        default="../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/")

    arg_parser.add_argument(
        "--neg-path",
        help="Path to the directory with the data of the negative patients",
        default="../../../datasets/BIMCV-COVID19-cIter_1_2-Negative/covid19_neg/")

    arg_parser.add_argument(
        "--common-ids",
        help="Path to the TSV with the subjects ID equivalence between the 'posi' and 'neg' datasets",
        default="../../../datasets/BIMCV-COVID19-cIter_1_2-Negative/listjoin_ok.tsv")

    arg_parser.add_argument(
        "--clean-data",
        help=("Path to the TSV file with the annotations generated with the "
              "data_cleaning.ipynb notebook to take the manually selected samples"),
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
        "--posi-target-labels",
        help=("Target labels to select the samples from the positive patients. "
              "The order is important, the first matching label will be taken "
              "as the label for the sample."),
        metavar="label_name",
        nargs='+',
        default=["COVID 19"],
        type=str)

    arg_parser.add_argument(
        "--neg-target-labels",
        help=("Target labels to select the samples from the negative patients. "
              "The order is important, the first matching label will be taken "
              "as the label for the sample."),
        metavar="label_name",
        nargs='+',
        default=["normal"],
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
        default=[0.7, 0.15, 0.15],
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
        "--avoid-preproc",
        help=("If set the script avoids any type of image preprocessing and "
              "only creates the final YAML file with the paths to the "
              "original images"),
        action="store_true")

    arg_parser.add_argument(
        "--images-to-avoid",
        help=("Path to a txt file with a list of images paths to avoid"),
        default=None,
        type=str)

    arg_parser.add_argument(
        "--preproc-path",
        help="Path to the folder to store the preprocessed dataset",
        default="../../../datasets/preproc_BIMCV-COVID19",
        type=str)

    arg_parser.add_argument(
        "--yaml-name",
        help=("Name of the YAML file to create (without the file extension). "
              "This name is also used to create the folder for the "
              "preprocessed images ('{YAML_NAME}_preproc_data')"),
        default="ecvl_bimcv_covid19",
        type=str)

    arg_parser.add_argument(
        "--train-balancing-type",
        help="Declares strategy to balance the samples in the training split",
        default=None,
        choices=["undersampling", "oversampling"],
        type=str)

    arg_parser.add_argument(
        "--avoid-balancing",
        help=("To disable the data balancing for all the splits. By default "
              "the validation and test splits are balanced."),
        action="store_true")

    arg_parser.add_argument(
        "--to-rgb",
        help="To convert the images from grayscale to RGB using a colormap",
        action="store_true")

    arg_parser.add_argument(
        "--colormap",
        help=("Name of the colormap to use for colorizing the images. "
              "It must be a colormap from matplotlib"),
        default='jet',
        type=str)

    arg_parser.add_argument(
        "--n-colors",
        help="Number of colors to use from the selected colormap",
        default=100,
        type=int)

    arg_parser.add_argument(
        "--only-dx",
        help=("Only select the images that are DX. "
              "Not available with --only-cr at the same time"),
        action="store_true")

    arg_parser.add_argument(
        "--only-cr",
        help=("Only select the images that are CR. "
              "Not available with --only-dx at the same time"),
        action="store_true")

    main(arg_parser.parse_args())
