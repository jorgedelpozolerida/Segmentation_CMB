#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Utility functions to plot project related data


{Long Description of Script}


@author: jorgedelpozolerida
@date: 05/11/2023
"""

import os
import sys
import argparse
import traceback

import matplotlib.pyplot as plt
import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import matplotlib.colors as mcolors


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def plot_mask_on_image(img_data, mask_data, cm_coords, alpha=0.5, title="",
                        axes_to_display=['sagittal', 'coronal', 'axial'], 
                        interpolation='none', cmap_custom=None):
    """
    Plot a mask on the corresponding image along specified axes, always displaying the axial view without a mask on the rightmost side.
    :param img_data: The 3D numpy array of the image data.
    :param mask_data: The 3D numpy array of the mask data.
    :param cm_coords: The (x, y, z) coordinates for the center of mass of the mask.
    :param alpha: The transparency of the mask overlay.
    :param title: The title of the plot.
    :param axes_to_display: A list of strings indicating which axes to display.
    :param interpolation: The interpolation method for displaying images.
    """
    
    # Ensure that both the image and the mask have the same shape
    assert img_data.shape == mask_data.shape, "Image and mask have different shapes!"
    
    # Find the center of mass of the mask
    x, y, z = int(cm_coords[0]), int(cm_coords[1]), int(cm_coords[2])
    
    # Determine the number of axes to display plus one for the axial without mask
    num_axes = len(axes_to_display) + 1
    fig, axes = plt.subplots(1, num_axes, figsize=(5 * num_axes, 5))
    
    # Create a color map for bright red overlay
    if cmap_custom is None:
        colors = [(0,0,0,0), (1,0,0,alpha)]  # from transparent to bright red
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', colors, N=2)
    else:
        cmap = cmap_custom

    # Dictionary to hold plane plotting functions
    plane_funcs = {
        'sagittal': lambda a: (a.imshow(img_data[x, :, :], cmap='gray'),
                                a.imshow(mask_data[x, :, :], alpha=alpha, cmap=cmap, interpolation=interpolation),
                                a.set_title(f"Sagittal (x={x})")),
        'coronal': lambda a: (a.imshow(img_data[:, y, :], cmap='gray'),
                                a.imshow(mask_data[:, y, :], alpha=alpha, cmap=cmap, interpolation=interpolation),
                                a.set_title(f"Coronal (y={y})")),
        'axial': lambda a: (a.imshow(img_data[:, :, z], cmap='gray'),
                            a.imshow(mask_data[:, :, z], alpha=alpha, cmap=cmap, interpolation=interpolation),
                            a.set_title(f"Axial (z={z})"))
    }
    
    # Plot each requested plane
    for i, axis in enumerate(axes_to_display):
        if axis in plane_funcs:
            plane_funcs[axis](axes[i])
    
    # Always display axial plane without mask on the rightmost side
    axes[-1].imshow(img_data[:, :, z], cmap='gray')
    axes[-1].set_title("Axial (no mask)")

    # Set the overall title and adjust the layout
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()