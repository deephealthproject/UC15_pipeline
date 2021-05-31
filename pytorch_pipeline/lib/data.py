"""
Module with the classes for creating the data generator for training.
"""
import os
import ast

import torch
import pytorch_lightning as pl
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image


#####################
# Data Augmentation #
#####################

def augmentations_v0_0(size: tuple) -> A.Compose:
    """
    Returns the v0.0 augmentations for the train split.
    The v0.0 is just for resizing the data, not to perform data augmentation.
    """
    return A.Compose([A.Resize(*size, interpolation=cv2.INTER_CUBIC),
                      A.ToFloat(255.0)])


def get_augmentations(version: str, size: tuple) -> A.Compose:
    """
    Auxiliary function to get the selected set of data augmentations using
    the Albumentations library.

    Args:
        version: A string with the version tag of the augmentations to select.

        size: A tuple (height, width) with the size to resize the images.

    Returns:
        A ecvl.DatasetAugmentations object that defines the augmentations for
        each split of the dataset.
    """
    if version == "0.0":
        return augmentations_v0_0(size)
    """
    if version == "1.0":
        return augmentations_v1_0(size)
    if version == "1.1":
        return augmentations_v1_1(size)
    """
    raise Exception("Wrong augmentations version provided!")


###################
# Data Generators #
###################

class COVIDDataset(torch.utils.data.Dataset):
    """
    Class to load the samples of the dataset.
    """

    def __init__(self, data_df, transformations=None):
        """
        Prepares the dataset object.

        Args:
            data_df: DataFrame with the samples data (image path, labels, ...).

            transformations: Transform object to apply to the images.
        """
        self.data_df = data_df
        self.trans = transformations

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data_df.index)

    def __getitem__(self, idx):
        """Returns one sample of the dataset selected by index."""
        sample_row = self.data_df.iloc[idx]
        img = np.array(Image.open(sample_row['filepath']))
        if self.trans:
            img = self.trans(image=img)["image"]

        return img, sample_row['labels']


class COVIDDataModule(pl.LightningDataModule):
    def __init__(self,
                 df_path,
                 target_labels,
                 batch_size=32,
                 shuffle=True,
                 target_size=(256, 256),
                 augmentations="0.0",
                 num_workers=0,
                 pin_memory=True):
        """
        Initializes the DataModule config.

        Args:
            df_path: Path to the DataFrame that stores the dataset data.

            target_labels: Target labels to select the samples for training.
                           The order is important, the first matching label
                           will be taken as the label for the sample.

            batch_size: Number of samples in each loaded batch. This applies
                        to the three splits (train, validation and test).

            shuffle: To shuffle the train split every epoch.

            target_size: Tuple of Height and Width to resize the images.

            augmentations: Set of transformations to use for data augmentation.
                           Selected by a string with a version tag.

            num_workers: Number of processes to use for data loading.

            pin_memory: To use pinned memory to load the data into the GPUs.
        """
        super().__init__()
        self.df_path = df_path
        self.target_labels = target_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size
        self.augs_version = augmentations
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _to_one_hot(self, label_str):
        """Converts a label string to the corresponding one hot encoding."""
        return np.array([int(label_str == l) for l in self.target_labels])

    def _select_label(self, labels_str):
        """
        Given a string representing a list of labels. Returns the label
        selected as the main label of the sample to classify.
        """
        # Get the list of labels from the string
        labels = ast.literal_eval(labels_str)
        # Get one target label from the list of labels
        for label in self.target_labels:
            if label in labels:
                return label

        raise Exception(f"No target label found in {labels}")

    def setup(self, stage=None):
        """Prepares the Datasets to load the samples."""
        # Load the samples data
        data_df = pd.read_csv(self.df_path, sep="\t")

        # Prepare the paths to the samples to be relative to the current dir
        subjects_path = os.path.dirname(self.df_path)
        data_df['filepath'] = data_df['filepath'].apply(
            lambda path: os.path.join(subjects_path, path))

        # Select the labels to classify each sample
        data_df['labels'] = data_df['labels'].apply(self._select_label)
        # Convert the labels to one hot encoding
        data_df['labels'] = data_df['labels'].apply(self._to_one_hot)

        # Prepare the Dataset object for the selected splits
        if stage in (None, 'fit'):
            train = data_df[data_df["split"] == "training"]
            # Get the transformation function to apply DA
            trans = get_augmentations(self.augs_version, self.target_size)
            self.train_dataset = COVIDDataset(train, trans)

        if stage in (None, 'validate'):
            val = data_df[data_df["split"] == "validation"]
            no_da_trans = get_augmentations("0.0", self.target_size)
            self.val_dataset = COVIDDataset(val, no_da_trans)

        if stage in (None, 'test'):
            test = data_df[data_df["split"] == "test"]
            no_da_trans = get_augmentations("0.0", self.target_size)
            self.test_dataset = COVIDDataset(test, no_da_trans)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)
