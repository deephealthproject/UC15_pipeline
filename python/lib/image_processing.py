"""
Module with auxiliary functions to deal with the images. From IO operations
to processing functions.
"""
import numpy as np
from PIL import Image
import nibabel as nib


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
