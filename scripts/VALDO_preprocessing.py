#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to preprocess  "VALDO challenge (Task2)" dataset

Steps and order followed to process data:

0. Quality Control of rawdata
    - handle Nans
    - mask preparation: binarize to same values masks
1. Skull Stripping
2. Crop (using brain mask)
3. Bias field correction?
4. Resampling and standarizing dimensions
5. Intensity normalization
6. Stack data together (multi-sequence input)

Notes:
- Works in paralell using as many CPUs as specified
- Registration was already performed in dataet, so not done


TODO:
- Quality check
- Bias field correction?
- Int Normalization
- multiprocessing  
- Create function to also return coordinates of center of microbleeds and how 
many out of brain masks and store as output of processing. Useful in the future for
visualization.

@author: jorgedelpozolerida
@date: 04/10/2023
"""


import os
import sys
import argparse
import traceback


import logging                                                                      
import numpy as np                                                                  
import pandas as pd                                                                 
import shutil
from tqdm import tqdm
import csv
import copy
import nibabel as nib
import re
import multiprocessing
import time 
from nilearn.image import resample_to_img, resample_img
from scipy.ndimage import generate_binary_structure, binary_closing, binary_dilation
import subprocess
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.filters import threshold_otsu
from datetime import datetime

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def ensure_directory_exists(dir_path):
    """ Create directory if non-existent """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def write_to_log_file(msg, log_file_path):
    '''
    Writes message to the log file.
    Args:
        msg (str): Message to be written to log file.
        log_file_path (str): Path to log file.
    '''
    current_time = datetime.now()
    with open(log_file_path, 'a+') as f:
        f.write('\n{}\n{}'.format(current_time, msg))

def get_largest_cc(segmentation):
    """
    Gets the largest connected component in image.
    Args:
        segmentation (np.ndarray): Image with blobs.
    Returns:
        largest_cc (np.ndarray): A binary image containing nothing but the largest
                                    connected component.
    """
    labels = label(segmentation)
    bincount = np.array(np.bincount(labels.flat))
    ind_large = np.argmax(bincount)  # Background is initially largest
    bincount[ind_large] = 0  # Remove background
    ind_large = np.argmax(bincount)  # This should now be largest connected component
    largest_cc = labels == ind_large

    return np.double(largest_cc)

def norm_min_max(input_image):
    '''
    Rescales intensities channel-wise to (25, 99) percentile.
    Args:
        input_image (np.ndarray): Input image array.
    Returns:
        image (np.ndarray): Image with rescaled intensities.
    '''
    image = copy.copy(input_image)

    for i in range(image.shape[-1]):
        if np.std(image[..., i]) != 0:
            p25, p99 = np.percentile(image[..., i], (25, 99))

            min_m = p25
            max_m = p99
            image[..., i] = rescale_intensity(image[..., i], in_range=(min_m, max_m))

    return image

def load_subject_data(args, subject):
    
    subject_old_dir = os.path.join(args.input_dir, subject)
    
    # Load NIfTI images for sequences using nibabel
    sequences = {
        "T1": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T1.nii.gz")),
        "T2": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T2.nii.gz")),
        "T2S": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T2S.nii.gz"))
    }
    
    # Load NIfTI images for labels using nibabel
    labels = {
        "T2S": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_CMB.nii.gz")) # bcs annotated in T2s
    }

    return sequences, labels

def resample(source_image, target_image, interpolation, is_annotation=False,
            isotropic=False, source_sequence=None, target_sequence=None, msg=''):
    '''
    Resamples source image to target image (no registration is performed).
    Args:
        source_image (nib.Nifti1Image): Source image being resampled.
        target_image (nib.Nifti1Image): Target image to which source is being resampled to.
        interpolation (str): Resampling method (one of nearest, linear and continuous).
        is_annotation (bool): Whether the source image is an annotation.
        isotropic (bool): Whether to resample to isotropic (uses only source image).
        source_sequence (str)(optional): Source sequence (for logging purposes only).
        target_sequence (str)(optional): Target sequence (for logging purposes only).
        msg (str)(optional): Log message.
    Returns:
        resampled_image (nib.Nifti1Image): Resampled source image.
        msg (str): Log message.
    '''
    if isotropic:
        msg += '\tResampling {} MRI to isotropic using {} interpolation...\n'.format(
            source_sequence, interpolation)
        msg += '\t\tShape before resampling: {}\n'.format(source_image.shape)

        resampled_image = resample_img(source_image, target_affine=np.eye(3),
                                        interpolation=interpolation,
                                        fill_value=np.min(source_image.get_fdata()),
                                        order='F')

    elif is_annotation:
        msg += '\tResampling {} annotation to {} using {} interpolation...\n'.format(
            source_sequence, target_sequence, interpolation)
        msg += '\t\tShape before resampling: {}\n'.format(source_image.shape)

        if interpolation == 'nearest':
            resampled_image = resample_to_img(source_image, target_image,
                                                interpolation=interpolation,
                                                ill_value=0)

        elif interpolation == 'linear':
            resampled_image = np.zeros(target_image.shape, dtype=np.float32)

            unique_labels = np.rint(np.unique(source_image.get_fdata()))
            for unique_label in unique_labels:

                annotation_binary = nib.Nifti1Image(
                    (np.rint(source_image.get_fdata()) == unique_label).astype(np.float32),
                    source_image.affine, source_image.header)

                annotation_binary = resample_to_img(annotation_binary, target_image,
                                                    interpolation=interpolation, fill_value=0)

                resampled_image[annotation_binary.get_fdata() >= 0.5] = unique_label

            resampled_image = nib.Nifti1Image(resampled_image, affine=annotation_binary.affine,
                                                header=annotation_binary.header)

    else:
        msg += '\tResampling {} MRI to {} using {} interpolation...\n'.format(
            source_sequence, target_sequence, interpolation)
        msg += '\t\tShape before resampling: {}\n'.format(source_image.shape)

        resampled_image = resample_to_img(source_image, target_image, interpolation=interpolation,
                                            fill_value=np.min(source_image.get_fdata()))

    msg += '\t\tShape after resampling: {}\n'.format(resampled_image.shape)

    return resampled_image, msg

def resample_mris_and_annotations(mris, annotations, primary_sequence, isotropic, msg=''):
    '''
    Resamples MRIs and annotations to primary sequence space.
    Args:
        mris (dict): Dictionary of MRIs.
        annotations (dict): Dictionary of annotations.
        primary_sequence (str): Sequence to which other sequences are being resampled to.
        isotropic (bool): Whether to resample to isotropic (uses only source image).
        msg (str)(optional): Log message.
    Returns:
        mris (dict): Dictionary of resampled MRIs.
        annotations (dict): Dictionary of resampled annotations.
        msg (str): Log message.
    '''
    msg += '\tResampling MRIs and annotations maps...\n'

    start = time.time()

    if isotropic:
        mris[primary_sequence], msg = resample(source_image=mris[primary_sequence],
                                                target_image=None,
                                                interpolation='linear',
                                                isotropic=True,
                                                source_sequence=primary_sequence,
                                                target_sequence=primary_sequence, msg=msg)
    for sequence in mris:
        # resample MRI
        if sequence != primary_sequence:
            mris[sequence], msg = resample(source_image=mris[sequence],
                                            target_image=mris[primary_sequence],
                                            interpolation='continuous',
                                            source_sequence=sequence,
                                            target_sequence=primary_sequence, msg=msg)
        # resample annotation
        annotations[sequence], msg = resample(source_image=annotations[sequence],
                                                target_image=mris[primary_sequence],
                                                interpolation='linear',
                                                is_annotation=True,
                                                source_sequence=sequence,
                                                target_sequence=primary_sequence, msg=msg)

    end = time.time()
    msg += '\t\tResampling of MRIs and annotations took {} seconds!\n\n'.format(end-start)

    return mris, annotations, msg

def get_brain_mask(image):
    """
    Computes brain mask using Otsu's thresholding and morphological operations.
    Args:
        image (nib.Nifti1Image): Primary sequence image.
    Returns:
        mask (np.ndarray): Computed brain mask.
    """
    # TODO: investigate if this a good fit for brain mask in all cases. Play around
    image_data = image.get_fdata()
    
    # Otsu's thresholding
    threshold = threshold_otsu(image_data)
    mask = image_data > threshold

    # Apply morphological operations
    struct = generate_binary_structure(3, 2)  # this defines the connectivity
    mask = binary_closing(mask, structure=struct)
    mask = get_largest_cc(mask)
    mask = binary_dilation(mask, iterations=5, structure=struct)

    return mask


def crop_and_concatenate(mris, annotations, primary_sequence, save_sequence_order, msg=''):
    '''
    Crops and concatenates MRIs and annotations to non-zero region.
    Args:
        mris (nib.Nifti1Image): Input MRIs.
        annotations (nib.Nifti1Image): Input annotations.
        primary_sequence (str): Sequence to which other sequences are being resampled to.
        save_sequence_order ([str]): Save sequence order.
        msg (str)(optional): Log message.
    Returns:
        cropped_mris (np.ndarray): Cropped MRIs array.
        cropped_annotations (np.ndarray): Cropped annotations array.
        msg (str): Log message.
    '''
    msg += '\tCropping and concatenating MRIs and annotations...\n'

    start = time.time()

    # get brain mask from primary sequence
    mask = get_brain_mask(image=mris[primary_sequence])

    x, y, z = np.where(mask == 1)
    coordinates = {'x': [np.min(x), np.max(x)], 'y': [np.min(y), np.max(y)],
                    'z': [np.min(z), np.max(z)]}

    # concatenate MRIs and annotations
    mris_array, annotations_array = [], []

    for sequence in save_sequence_order:
        mris_array.append(mris[sequence].get_fdata()[..., None])
        annotations_array.append(annotations[sequence].get_fdata()[..., None])

    mris_array = np.concatenate(mris_array, axis=-1)
    annotations_array = np.concatenate(annotations_array, axis=-1)

    msg += '\t\tMRIs shape after concatenation: {}\n'.format(mris_array.shape)
    msg += '\t\tAnnotations shape after concatenation: {}\n'.format(annotations_array.shape)

    # crop MRIs and annotations by applying brain mask
    cropped_mris = mris_array[coordinates['x'][0]:coordinates['x'][1],
                                coordinates['y'][0]:coordinates['y'][1],
                                coordinates['z'][0]:coordinates['z'][1], :]

    cropped_annotations = annotations_array[coordinates['x'][0]:coordinates['x'][1],
                                            coordinates['y'][0]:coordinates['y'][1],
                                            coordinates['z'][0]:coordinates['z'][1], :]

    msg += '\t\tMRIs shape after cropping: {}\n'.format(cropped_mris.shape)
    msg += '\t\tAnnotations shape after cropping: {}\n'.format(cropped_annotations.shape)

    end = time.time()
    msg += '\t\tCropping and concatenation of MRIs and annotations took {} seconds!\n\n'.format(end-start)  # NOQA E501

    return cropped_mris, cropped_annotations, msg

def process_study(args, subject, msg=''):
    
    msg = '' # handle this later on, for logging purposes
    # Handle old and new paths
    subject_new_dir = os.path.join(args.output_dir, subject)

    subject_labels_dir = os.path.join(subject_new_dir, "Labels")
    subject_niftis_dir = os.path.join(subject_new_dir, "Nifti")
    
    ensure_directory_exists(subject_labels_dir)
    ensure_directory_exists(subject_niftis_dir)

    # Load data from subject
    mris, annotations = load_subject_data(args, subject)
    
    # 0. Quality Control of rawdata TODO
    #     - handle Nans
    #     - mask preparation: binarize to same values masks
    
    # 1. Skull Stripping  -> NOTE: seems to be done already in dataset, not necessary

    # 2. Crop (using brain mask)
    
    mris_array, annotations_array, msg = crop_and_concatenate(
        mris, annotations, primary_sequence=args.primary_sequence, save_sequence_order = args.save_sequence_order, msg=msg)

    # 3. Bias field correction? TODO
    
    # 4. Resampling and standarizing dimensions (isotropic)
    mris, annotations, msg = resample_mris_and_annotations(mris, annotations, primary_sequence=args.primary_sequence, isotropic = True)
    
    # save affine of the resampled primary sequence
    affine_after_resampling = mris[args.primary_sequence].affine
    header_after_resampling = mris[args.primary_sequence].header
    
    
    # 5. Intensity normalization TODO
    
    
    # 6. Stack data together (multi-sequence input)


    # -------------------------------------------------------------
    
    # save MRIs and annotations as Nifti1Image
    mris_image = nib.Nifti1Image(mris_array.astype(np.float32), 
                                affine=affine_after_resampling,
                                header=header_after_resampling)
    mris_image.set_data_dtype(np.float32)

    annotations_image = nib.Nifti1Image(annotations_array.astype(np.uint8),
                                        affine=affine_after_resampling,
                                        header=header_after_resampling)
    annotations_image.set_data_dtype(np.uint8)

    return mris_image, annotations_image, msg


def main(args):

    args.dataset_dir_path = os.path.join(args.output_dir, args.dataset_name)
    data_dir_path = os.path.join(args.dataset_dir_path, 'Data')
    args.mris_dir_path = os.path.join(data_dir_path, 'MRIs')
    args.annotations_dir_path = os.path.join(data_dir_path, 'Annotations')
    for dir_p in [args.output_dir, args.dataset_dir_path, args.mris_dir_path, args.annotations_dir_path ]:
        print(dir_p)
        ensure_directory_exists(dir_p)

    current_time = datetime.now()

    current_datetime = current_time.strftime("%d%m%Y_%H%M%S")
    args.log_file_path = os.path.join(args.dataset_dir_path, 'log_' + current_datetime + '.txt')
    
    
    # Ignore folder starting with weird symbol
    # subjects = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    subjects = ["sub-101", "sub-102"]

    print(subjects)
    
    for subject in subjects:
        # initialize timer
        start = time.time()

        # initialize log message
        msg = 'Started processing {}...\n\n'.format(subject)

        try:
            # ------------
            mris, annotations, msg = process_study(args, subject, msg='')
            # save MRIs
            mris_path = os.path.join(args.output_dir, "MRIs", subject + '.nii.gz')
            nib.save(mris, mris_path)

            # save annotations
            annotations_path = os.path.join(args.annotations_dir_path, "Annotations", subject + '.nii.gz')
            nib.save(annotations, annotations_path)

        except Exception:
            msg += 'Failed to process {}!\n\nException caught: {}'.format(
                subject, traceback.format_exc())

        end = time.time()
        msg += 'Finished processing of {} in {} seconds!\n\n'.format(
            subject, end-start)

        write_to_log_file(msg, args.log_file_path)

def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_sequence_order', nargs='+', type=str,
                        default=['T2S', 'T2', 'T1'],
                        help='Order in which sequences in output MRI image will be saved')
    parser.add_argument('--primary_sequence', type=str, default='T2S',
                        help='Primary sequence (to which the rest will be conformed to)')
    parser.add_argument('--input_dir', type=str, default=None, required=True,
                        help='Path to the input directory of dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Full path to the directory where processed images will be saved')
    parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of workers running in parallel')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Name of the dataset')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
    