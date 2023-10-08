#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing functions for visualization


{Long Description of Script}


@author: jorgedelpozolerida
@date: 04/10/2023
"""


import os
import sys
import argparse
import traceback


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import nibabel as nib
from scipy.ndimage import center_of_mass
from nilearn import plotting
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)



def visualize_CMB_overlayed(mri_im, mask_im):
    """
    Visualize overlayed picture by getting slice 
    at the center of mass of the mask.
    


    Returns:
        - coords: list, coordinates of the center of mass
    """
    
    
    # Get the mask data and ensure it's binary (0 and 1)
    mask = mask_im.get_fdata()
    mask = (mask > 0).astype(int)

    # Calculate the center of mass
    com_y, com_x, com_z = center_of_mass(mask)
    coords = [com_x, com_y, com_z]

    # Plotting the original image with the mask overlay at the center of mass
    plotting.plot_roi(mask_im, mri_im, cut_coords=coords, display_mode='ortho', colorbar=True,
                      title='CMB segmentation', alpha=0.5, cmap='autumn')


    plotting.show()

    return coords



def plot_mask_on_image(img_nii, mask_nii):
    """Plot 3D mask on top of a 3D image using the center of mass of the mask."""
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
    # Ensure that both the image and the mask have the same shape
    assert img_data.shape == mask_data.shape, "Image and mask have different shapes!"
    
    # Find the center of mass of the mask
    center_of_mass = np.round(np.array(np.nonzero(mask_data)).mean(axis=1)).astype(int)
    x, y, z = center_of_mass
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot slices from each dimension at the center of mass
    axes[0].imshow(img_data[x, :, :], cmap='gray')
    axes[0].imshow(mask_data[x, :, :], alpha=0.5, cmap='Reds', interpolation='none')
    axes[0].set_title(f"Sagittal plane at x={x}")

    axes[1].imshow(img_data[:, y, :], cmap='gray')
    axes[1].imshow(mask_data[:, y, :], alpha=0.5, cmap='Reds', interpolation='none')
    axes[1].set_title(f"Coronal plane at y={y}")

    axes[2].imshow(img_data[:, :, z], cmap='gray')
    axes[2].imshow(mask_data[:, :, z], alpha=0.5, cmap='Reds', interpolation='none')
    axes[2].set_title(f"Axial plane at z={z}")

    plt.tight_layout()
    plt.show()