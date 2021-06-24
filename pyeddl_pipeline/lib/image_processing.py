"""
Module with auxiliary functions to deal with the images. From IO operations
to processing functions.
"""
import os
import json

import numpy as np
from PIL import Image
import nibabel as nib
from skimage import exposure
import albumentations as A
from matplotlib import pyplot as plt


def png2numpy(png_path: str) -> np.ndarray:
    """
    Extracts the image pixels data from a png file.

    Args:
        png_path: Path to the png file to extract data from.

    Returns:
        A numpy array with the image.
    """
    pil_img = Image.open(png_path)
    return np.array(pil_img)


def nifty2numpy(nifti_path: str) -> np.ndarray:
    """
    Extracts the image pixels data from a nifti file.

    Args:
        nifti_path: Path to the nifti file to extract data from.

    Returns:
        A numpy array with the image.
    """
    nifti_img = nib.load(nifti_path)
    return nifti_img.get_fdata()


def load_numpy_data(file_path: str) -> np.ndarray:
    """
    Given a path to a .png or .nii.gz file returns the corresponing image data
    in a numpy array.

    Args:
        file_path: Path to the file to extract data from.

    Returns:
        A numpy array with the image.
    """
    if file_path.endswith(".png"):
        return png2numpy(file_path)

    if file_path.endswith(".nii.gz"):
        return nifty2numpy(file_path)

    raise NameError(f'The extension of the file "{file_path}" is not valid!')


def get_stats(img_file_path: str) -> list:
    """
    Computes some statistics of the given image.

    Args:
        img_file_path: Path to the image to load (.png or .nii.gz format).

    Returns:
        A list with some stats of the image.
        [pixels_mean, pixels_std, maximum_pixel, minimum_pixel, img_shape]
    """
    img = load_numpy_data(img_file_path)
    return [img.mean(), img.std(), img.max(), img.mean(), img.shape]


def to_rgba_with_cmap(img: np.ndarray,
                      colormap: str = 'jet',
                      n_colors: int = 100):
    """
    Using the colormap specified, converts the input image to a colored
    RGB image.

    Args:
        colormap: Colormap to use. It must be a color map from matplotlib.

        n_colors: Number of colors to use from the colormap.

    Returns:
        A numpy array with the colored image.
    """
    cmap = plt.get_cmap(colormap, lut=n_colors)  # Prepare the colormap
    return cmap(img)  # Apply the colorization


def preprocess_image(img_path: str,
                     img_outpath: str,
                     hist_type: str = "adaptive",
                     invert_img: bool = True,
                     to_rgb: bool = False,
                     colormap: str = 'jet',
                     n_colors: int = 100):
    """
    Applies histogram equalization to the image given and stores the
    preprocessed version in the provided output path. Other transformations
    can be applied if activated: invert grayscale colors or convert to RGB
    with a colormap from matplotlib.

    Args:
        img_path: Path to the image to preprocess (it must be a .png).

        img_outpath: Path of the output .png file to create.

        hist_type: Type of histogram equalization.
                   It can be "normal" or "adaptive".

        invert_img: If True, checks the image metadata to see if the image
                    values should be inverted. The metadata should be in a
                    json file with the same name than the image and it must
                    be in the same folder.
    """
    img = load_numpy_data(img_path)

    if invert_img:
        # The images that are monochrome 1 must be inverted
        json_path = img_path[:-4] + ".json"
        if os.path.isfile(json_path):
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                if data["00280004"]["Value"][0] == "MONOCHROME1":
                    print(f"Going to invert image '{img_path}'")
                    img = img.max() - img  # Invert the image
        else:
            print(f"JSON with metadata not found for image '{img_path}'")

    # Apply equalization
    if hist_type == "adaptive":
        img = exposure.equalize_adapthist(img)
    elif hist_type == "normal":
        img = exposure.equalize_hist(img)
    else:
        raise Exception("Wrong histogram equalization type provided!")

    if to_rgb:
        # Apply colorization. Te output is RGBA (H, W, 4)
        img = to_rgba_with_cmap(img, colormap, n_colors)

    # After histogram equalization the values are floats in the range [0-1].
    # Convert the images to the range [0-255] with uint8 values
    img = np.uint8(img * 255.0)

    # From numpy to PIL image
    if to_rgb:
        pil_img = Image.fromarray(img, mode='RGBA')
        pil_img = pil_img.convert('RGB')
    else:
        pil_img = Image.fromarray(img, mode='L')

    # Store the processed image
    pil_img.save(img_outpath, format="PNG")


def create_copy_with_DA(img_path: str,
                        img_outpath: str):
    """
    Creates a copy of an images with data augmentation applied.

    Args:
        img_path: Path to the image to copy (it must be a .png).

        img_outpath: Path of the output .png file to create.
    """
    img = load_numpy_data(img_path)
    n_channels = img.shape[-1]

    # DA operations
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2,
                                   always_apply=True),
        A.Affine(scale=[0.9, 1.1],
                 translate_percent=0.05,
                 rotate=[-10, 10],
                 always_apply=True)
    ])

    # Apply the transformations
    aug_img = np.uint8(transform(image=img)["image"])

    # Store the transformed copy
    pil_img = Image.fromarray(aug_img, mode='RGB' if n_channels == 3 else 'L')
    pil_img.save(img_outpath, format="PNG")
