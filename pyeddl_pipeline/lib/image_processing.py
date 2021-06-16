"""
Module with auxiliary functions to deal with the images. From IO operations
to processing functions.
"""
import numpy as np
from PIL import Image
import nibabel as nib
from skimage import exposure
import albumentations as A


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


def histogram_equalization(img_path: str,
                           img_outpath: str,
                           hist_type: str = "adaptive"):
    """
    Applies histogram equalization to the image given and stores the
    preprocessed version in the provided output path.

    Args:
        img_path: Path to the image to preprocess (it must be a .png).

        img_outpath: Path of the output .png file to create.

        hist_type: Type of histogram equalization.
                   It can be "normal" or "adaptive".
    """
    img = load_numpy_data(img_path)

    # Apply equalization
    if hist_type == "adaptive":
        img = exposure.equalize_adapthist(img)
    elif hist_type == "normal":
        img = exposure.equalize_hist(img)
    else:
        raise Exception("Wrong histogram equalization type provided!")

    # After histogram equalization the values are floats in the range [0-1].
    # Convert the images to the range [0-255] with uint8 values
    img = np.uint8(img * 255.0)

    # Store the processed image
    pil_img = Image.fromarray(img, mode='L')
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
    pil_img = Image.fromarray(aug_img, mode='L')
    pil_img.save(img_outpath, format="PNG")
