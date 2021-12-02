"""
Module with the classes for creating the data generators and the utilities to
perform data augmentation.
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


def augmentations_v1_0(size: tuple) -> A.Compose:
    """
    Returns the v1.0 augmentations for the train split.
    The v1.0 applies some basic augmentations playing with the brightness,
    contrast and small image rotations.
    """
    return A.Compose([A.Resize(*size, interpolation=cv2.INTER_CUBIC),
                      A.HorizontalFlip(p=0.5),
                      A.Rotate(limit=(-10, 10)),
                      A.RandomBrightnessContrast(brightness_limit=(0, 0.2),
                                                 contrast_limit=0.2),
                      A.ToFloat(255.0)])


def augmentations_v1_1(size: tuple) -> A.Compose:
    """
    Returns the v1.1 augmentations for the train split.
    The v1.1 applies the same augmentations than the v1.0 but more
    aggressively.
    """
    return A.Compose([A.Resize(*size, interpolation=cv2.INTER_CUBIC),
                      A.HorizontalFlip(p=0.5),
                      A.Rotate(limit=(-15, 15)),
                      A.RandomBrightnessContrast(brightness_limit=(0, 0.3),
                                                 contrast_limit=0.4),
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
    if version == "1.0":
        return augmentations_v1_0(size)
    if version == "1.1":
        return augmentations_v1_1(size)

    raise Exception("Wrong augmentations version provided!")


###################
# Data Generators #
###################

class COVIDDataset(torch.utils.data.Dataset):
    """
    Class to load the samples of the dataset.
    """

    def __init__(self, data_df, transformations=None, to_rgb=False):
        """
        Prepares the dataset object.

        Args:
            data_df: DataFrame with the samples data (image path, labels, ...).

            transformations: Transform object to apply to the images.

            to_rgb: If true converts the grayscale images to rgb replicating
                    the single channel.
        """
        self.data_df = data_df
        self.trans = transformations
        self.to_rgb = to_rgb

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data_df.index)

    def __getitem__(self, idx):
        """Returns one sample of the dataset selected by index."""
        sample_row = self.data_df.iloc[idx]
        img = np.array(Image.open(sample_row['filepath']))
        if self.trans:
            img = self.trans(image=img)["image"]

        # Prepare the torch tensors
        img = torch.tensor(img)
        if self.to_rgb:
            img = img.repeat(3, 1, 1)  # Replicate the single channel
        else:
            img = img.view((1, *img.shape))  # Add the channel dimension

        label = torch.tensor(sample_row['labels'])

        return img, label


class COVIDDataModule(pl.LightningDataModule):
    def __init__(self,
                 df_path,
                 target_labels,
                 batch_size=32,
                 shuffle=True,
                 target_size=(256, 256),
                 augmentations="0.0",
                 num_workers=0,
                 pin_memory=True,
                 to_rgb=False,
                 drop_last=True):
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

            to_rgb: If true converts the grayscale images to rgb replicating
                    the single channel.

            drop_last: To drop the last batch incomplete batch, if the dataset
                       is not divisible by the batch size.
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
        self.to_rgb = to_rgb
        self.drop_last = drop_last

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
            self.train_dataset = COVIDDataset(train, trans, self.to_rgb)

        if stage in (None, 'validate', 'fit'):
            val = data_df[data_df["split"] == "validation"]
            no_da_trans = get_augmentations("0.0", self.target_size)
            self.val_dataset = COVIDDataset(val, no_da_trans, self.to_rgb)

        if stage in (None, 'test'):
            test = data_df[data_df["split"] == "test"]
            no_da_trans = get_augmentations("0.0", self.target_size)
            self.test_dataset = COVIDDataset(test, no_da_trans, self.to_rgb)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           drop_last=self.drop_last)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           drop_last=self.drop_last)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           drop_last=False)
