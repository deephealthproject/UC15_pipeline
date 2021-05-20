"""
Module with auxiliary functions for the data analysis
"""
import math
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def create_histogram(data: list,
                     title: str,
                     ylabel: str,
                     xlabel: str,
                     out_path: str = "",
                     bins: int = -1,
                     log_yscale: bool = True,
                     print_grid: bool = True,
                     fig_size: Tuple[int, int] = (5, 4),
                     dpi: int = 100,
                     verbose: bool = True):
    """
    Creates a histogram plot and shows or stores it to a .png file.

    Args:
        data: List of data to plot.

        title: Title string of the plot.

        ylabel: Label string for the y axis of the plot.

        xlabel: Label string for the x axis of the plot.

        out_path: Path to save the output .png file. If "" the plot is shown.

        bins: Number of bins in the range. If -1, bins is equal to the number
              of different values in "data".

        log_yscale: To use a logarithmic scale for the y axis.

        print_grid: To print a grid over the plot for better visualization.

        fig_size: Pyplot output figure size (width, height).

        dpi: dpi resolution of the output image.

        verbose: To enable printed logs.
    """
    plt.figure(figsize=fig_size, dpi=dpi)  # Set figure size

    if bins == -1:
        # If the bins are not provided we put as many bins as unique values
        bins = len(set(data))

    plt.hist(data, bins=bins)  # Create the histogram

    # Configure plot
    if log_yscale:
        plt.yscale('log')
    if print_grid:
        plt.grid()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    # Show or save the plot image
    if out_path == "":
        plt.show()
    else:
        plt.savefig(out_path, dpi=dpi)
        if verbose:
            print(f'Histogram "{title}" saved in: {out_path}')

    plt.clf()  # Clear figure for next plot


def show_images(images: list,
                rows: int = -1,
                columns: int = -1,
                grid_height: int = 10,
                grid_width: int = 10):
    """
    Given a list with images (numpy arrays) this function plots all the images
    of the list in a grid.

    Args:
        images: List of numpy arrays of the images.

        rows: Number of rows for the grid of images to show. If -1, defaults
              to the maximum number possible depending on the columns argument.

        columns: Number of columns for the grid of images to show. If -1,
                 defaults to 2.

        grid_height: Total height of the grid to plot.

        grid_width: Total width of the grid to plot.
    """
    # Prepare grid configuration
    n_images = len(images)
    if n_images == 0:
        print("Warning: 0 images provided to show_images()!")
        return
    elif n_images == 1:
        r, c = 1, 1
    elif rows == -1 and columns == -1:
        c = 2  # Default to 2 columns
        r = math.ceil(n_images / c)
    elif rows == -1:
        c = columns
        r = math.ceil(n_images / c)
    elif columns == -1:
        r = rows
        c = math.ceil(n_images / r)
    else:
        r, c = rows, columns
        assert r * c >= n_images, "The grid dimensions are too small!"

    # Initialize the grid of subplots
    fig, axes = plt.subplots(r, c, squeeze=False, constrained_layout=True)
    # Set grid size
    fig.set_figheight(grid_height)
    fig.set_figwidth(grid_width)

    # Fill the grid of images
    aux_r, aux_c = 0, 0  # To track the current grid postion
    for img in images:
        # Show the image in the current grid position
        axes[aux_r, aux_c].imshow(img, cmap="bone")

        # Update current grid postion
        #  - Note: Move first left-right, then top-bottom
        aux_c = (aux_c + 1) % c
        if aux_c == 0:
            aux_r += 1

    plt.show()  # Ensure that the grid is shown
