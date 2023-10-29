#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to pre-process datasets

Steps and order followed to process data:

# 1. Perform QC while loading data
# 2. Resample and Standardize
# 3. Crop (using brain mask)
# 4. Concatenate (stack together into single file)

Datasets implement:
- VALDO challenge

Notes:
- Works in paralell using as many CPUs as specified
- Registration was already performed in dataet, so not done
- Also retrieves and saves metadata for masks before and after processing

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
from scipy.ndimage import generate_binary_structure, binary_closing, binary_dilation, binary_erosion, center_of_mass
from scipy.ndimage import label as nd_label
import json
import subprocess
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.filters import threshold_otsu
from datetime import datetime
import pickle 
from functools import partial

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
        f.write(f'\n{current_time}\n{msg}')

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

def process_VALDO_mri(mri_im, msg=''):
    """
    Process a VALDO MRI image to handle NaNs by replacing them with the background value.

    Args:
    - mri_im (nibabel.Nifti1Image): The nibabel object of the MRI.

    Returns:
    - nibabel.Nifti1Image: Processed MRI as a nibabel object.
    - str: Updated log message.
    """
    
    # Extract data from nibabel object
    data = mri_im.get_fdata().copy()
    
    # Identify NaNs
    nan_mask = np.isnan(data)
    num_nans = np.sum(nan_mask)
    perc_nans = num_nans/len(data.flatten())*100
    
    if num_nans > 0:
        # Compute the background value using small patches from the edges
        edge_patches = [data[:10, :10, :5], data[-10:, :10, :5], 
                        data[:10, -10:, :5], data[-10:, -10:, :5], 
                        data[:10, :10, -5:], data[-10:, :10, -5:], 
                        data[:10, -10:, -5:], data[-10:, -10:, -5:]]
        background_value = np.nanmedian(np.concatenate(edge_patches))
        if np.isnan(background_value):
            background_value = 0
            msg += f'\t\tForced background value to 0 as region selected is full of nan\n'
        # Replace NaNs with the background value
        data[nan_mask] = background_value

        msg += f'\t\tFound {round(perc_nans, 2)}% of NaNs and replaced with background value: {background_value}\n'
    
    # Convert processed data back to Nifti1Image
    processed_mri_im = nib.Nifti1Image(data, mri_im.affine, mri_im.header)

    return processed_mri_im, msg

def process_cmb_mask(label_im, msg):
    """
    Process a nibabel object containing a mask of cerebral microbleeds (CMBs).

    Args:
        label_im (nibabel.Nifti1Image): The nibabel object of the mask.
        msg (str): Log message to be updated.

    Returns:
        processed_mask_nib (nibabel.Nifti1Image): Processed mask as a nibabel object.
        com_list (list[tuple]): List of centers of mass for each connected component.
        pixel_counts (list[int]): List of pixel counts for each connected component.
        radii (list[float]): List of equivalent radii for each connected component.
        msg (str): Updated log message.
    """

    # Extract data from the label image
    data = label_im.get_fdata()

    # Identify and handle unique labels in the mask
    unique_labels, counts = np.unique(data, return_counts=True)
    if len(unique_labels) > 2:
        raise ValueError("More than two unique labels found in the mask.")
    elif len(unique_labels) == 1:
        msg += f"\t\tOnly one label found: {unique_labels[0]}\n"
        data[:] = 0
    else:
        majority_label, minority_label = unique_labels[np.argmax(counts)], unique_labels[np.argmin(counts)]
        data[data == majority_label], data[data == minority_label] = 0, 1

    # Find connected components in the mask
    labeled_array, num_features = nd_label(data)

    # Calculate centers of mass and pixel counts
    com_list = center_of_mass(data, labels=labeled_array, index=np.arange(1, num_features + 1))
    pixel_counts = np.bincount(labeled_array.ravel())[1:]

    # Calculate radii assuming CMBs are spherical
    radii = [(3 * count / (4 * np.pi))**(1/3) for count in pixel_counts]

    # Convert the processed mask data back to a nibabel object
    processed_mask_nib = nib.Nifti1Image(data, label_im.affine, label_im.header)

    # Update the log message
    msg += f'\t\tNumber of CMBs: {len(com_list)}. Sizes: {pixel_counts},' + \
            f' Radii: {radii}, Unique labels: {unique_labels}, Counts: {counts}\n'

    metadata = {
        'centers_of_mass': com_list,
        'pixel_counts': pixel_counts,
        'radii': radii
    }

    return processed_mask_nib, metadata, msg



def perform_VALDO_QC(args, subject, mris, annotations, msg):
    """
    Perform Quality Control (QC) specific to the VALDO dataset on MRI sequences and labels.

    Args:
        args (Namespace): Arguments passed to the main function.
        subject (str): The subject identifier.
        mris (dict): Dictionary of MRI sequences.
        annotations (dict): Dictionary of labels.
        msg (str): Log message.

    Returns:
        mris_qc (dict): Dictionary of QC'ed MRI sequences.
        annotations_qc (dict): Dictionary of QC'ed labels.
        annotations_metadata (dict): Metadata associated with the QC'ed labels.
        msg (str): Updated log message.
    """

    mris_qc, annotations_qc, annotations_metadata = {}, {}, {}

    # Quality Control of Labels
    for anno_sequence, anno_im in annotations.items():
        annotations_qc[anno_sequence], metadata, msg = process_cmb_mask(anno_im, msg)
        annotations_metadata[anno_sequence] = metadata

    # Quality Control of MRI Sequences
    for mri_sequence, mri_im in mris.items():
        mris_qc[mri_sequence], msg = process_VALDO_mri(mri_im, msg)
    
    return mris_qc, annotations_qc, annotations_metadata, msg


def load_mris_and_annotations(args, subject, msg=''):
    '''
    Loads MRI scans and their corresponding annotations for a given subject 
    from a specific dataset and performs orientation fix.    
    
    Args:
        args (object): Contains configuration parameters, including input directory and dataset name.
        subject (str): Identifier of the subject whose MRI scans and annotations are to be loaded.
        msg (str, optional): A string for logging purposes. Default is an empty string.
        
    Returns:
        mris (dict): Dictionary where keys are sequence names (e.g., "T1", "T2") and values are 
                        the corresponding MRI scans loaded as nibabel.Nifti1Image objects.
        annotations (dict): Dictionary where keys are sequence names and values are the corresponding 
                            annotations loaded as nibabel.Nifti1Image objects.
        labels_metadata (dict): Dictionary containing metadata related to labels (annotations).
        msg (str): Updated log message.

    '''
    msg += '\tLoading MRI scans and annotations...\n'

    subject_old_dir = os.path.join(args.input_dir, subject)

    if "VALDO" in args.dataset_name:
        # Load NIfTI images for sequences using nibabel
        sequences_raw, labels_raw, labels_metadata,  msg = load_VALDO_data(subject, subject_old_dir, msg)
    else:
        # Implement here for other datasets
        raise NotImplemented
    
    start = time.time()

    mris = {}
    annotations = {}

    # Fill MRIs dict
    for sequence_name in sequences_raw:
        mris[sequence_name] = sequences_raw[sequence_name]
        msg += f'\t\tFound {sequence_name} MRI sequence of shape {mris[sequence_name].shape}\n'

        # fix orientation adn data type
        mris[sequence_name] = nib.as_closest_canonical(mris[sequence_name])
        mris[sequence_name].set_data_dtype(np.float32) 


    # Fill annotations dict
    for sequence_name in sequences_raw:
        if sequence_name in labels_raw.keys():
            annotations[sequence_name] = labels_raw[sequence_name]
            msg += f'\t\tFound {sequence_name} annotation of shape {annotations[sequence_name].shape}\n'
        else:
            annotations[sequence_name] = nib.Nifti1Image(np.zeros(shape=mris[args.primary_sequence].shape),
                                                    affine=mris[args.primary_sequence].affine,
                                                    header=mris[args.primary_sequence].header)
            msg += f'\t\tMissing {sequence_name} annotation, filling with 0s\n'

        # fix orientation adn data type
        annotations[sequence_name] = nib.as_closest_canonical(annotations[sequence_name])
        annotations[sequence_name].set_data_dtype(np.uint8)


    end = time.time()
    msg += f'\t\tLoading of MRIs and annotations took {end - start} seconds!\n\n'

    return mris, annotations, labels_metadata, msg

def load_VALDO_data(subject, subject_old_dir, msg):
    """
    Load MRI sequences and labels specific to the VALDO dataset. PErforms QC in the process.

    Args:
        subject (str): The subject identifier.
        subject_old_dir (str): Directory path where the subject's data is located.
        msg (str): Log message.

    Returns:
        sequences_qc (dict): Dictionary of QC'ed MRI sequences.
        labels_qc (dict): Dictionary of QC'ed labels.
        labels_metadata (dict): Metadata associated with the labels.
        msg (str): Updated log message.
    """
    
    # 1. Load Raw MRI Sequences
    sequences_raw = {
        "T1": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T1.nii.gz")),
        "T2": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T2.nii.gz")),
        "T2S": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T2S.nii.gz"))
    }
    
    # 2. Load Raw Labels (Annotations are made in T2S space for VALDO dataset)
    labels_raw = {
        "T2S": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_CMB.nii.gz"))
    }
    
    # 3. Perform Quality Control (QC) on Loaded Data
    sequences_qc, labels_qc, labels_metadata, msg = perform_VALDO_QC(args, subject, sequences_raw, labels_raw, msg)
    
    return sequences_qc, labels_qc, labels_metadata, msg


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
        msg += f'\tResampling {source_sequence} MRI to isotropic using {interpolation} interpolation...\n'
        msg += f'\t\tShape before resampling: {source_image.shape}\n'

        resampled_image = resample_img(source_image, target_affine=np.eye(3),
                                        interpolation=interpolation,
                                        fill_value=np.min(source_image.get_fdata()),
                                        order='F')

    elif is_annotation:
        msg += f'\tResampling {source_sequence} annotation to {target_sequence} using {interpolation} interpolation...\n'
        msg += f'\t\tShape before resampling: {source_image.shape}\n'

        if interpolation == 'nearest':
            resampled_image = resample_to_img(source_image, target_image,
                                                interpolation=interpolation,
                                                fill_value=0)

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
        msg += f'\tResampling {source_sequence} MRI to {target_sequence} using {interpolation} interpolation...\n'
        msg += f'\t\tShape before resampling: {source_image.shape}\n'

        resampled_image = resample_to_img(source_image, target_image, interpolation=interpolation,
                                            fill_value=np.min(source_image.get_fdata()))

    msg += f'\t\tShape after resampling: {resampled_image.shape}\n'

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
                                                # interpolation='nearest',
                                                interpolation='linear',
                                                is_annotation=True,
                                                source_sequence=sequence,
                                                target_sequence=primary_sequence, msg=msg)

    end = time.time()
    msg += f'\t\tResampling of MRIs and annotations took {end - start} seconds!\n\n'

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

    msg += f'\t\tMRIs shape after concatenation: {mris_array.shape}\n'
    msg += f'\t\tAnnotations shape after concatenation: {annotations_array.shape}\n'

    # crop MRIs and annotations by applying brain mask
    cropped_mris = mris_array[coordinates['x'][0]:coordinates['x'][1],
                                coordinates['y'][0]:coordinates['y'][1],
                                coordinates['z'][0]:coordinates['z'][1], :]

    cropped_annotations = annotations_array[coordinates['x'][0]:coordinates['x'][1],
                                            coordinates['y'][0]:coordinates['y'][1],
                                            coordinates['z'][0]:coordinates['z'][1], :]

    msg += f'\t\tMRIs shape after cropping: {cropped_mris.shape}\n'
    msg += f'\t\tAnnotations shape after cropping: {cropped_annotations.shape}\n'

    end = time.time()
    msg += f'\t\tCropping and concatenation of MRIs and annotations took {end - start} seconds!\n\n'

    return cropped_mris, cropped_annotations, msg

def combine_annotations(annotations, priorities, msg=''):
    '''
    Combines multi-channel annotations to single-channel according to label priotiries.
    Args:
        annotations (np.array): Annotations array.
        priorities ([int]): Label priorities.
        msg (str)(optional): Log message.
    Returns:
        combined_annotations (np.array): Combined annotations array.
        msg (optional): Log message.
    '''
    
    # TODO: if with future datasets several labels, combine here. 
    
    # For now let's just take first channel (T2S)
    combined_annotations = annotations[:, :, :, 0]
    
    return combined_annotations, msg

def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj

def process_study(args, subject, msg=''):
    """
    Process a given study (subject) by performing a series of operations 
    including loading, resampling, cropping, and saving the MRIs and annotations.
    
    Args:
        args (dict): Parsed arguments coming from parse_args() function.
        subject (str): The subject identifier.
        msg (str, optional): Log message. Defaults to ''.
    
    Returns:
        None. The function writes processed data to disk and updates the log message.
    """
    
    # Initialize
    start = time.time()
    msg = f'Started processing {subject}...\n\n'
    
    # Create dirs
    for sub_d in [args.mris_subdir, args.annotations_subdir, args.annotations_metadata_subdir]:
        ensure_directory_exists(os.path.join(args.data_dir_path, subject, sub_d))
    
    
    try:
        # 1. Perform QC while loading data
        mris, annotations, labels_metadata, msg = load_mris_and_annotations(args, subject, msg)
        
        # 2. Resample and Standardize
        mris, annotations, msg = resample_mris_and_annotations(mris, annotations, 
                                                                primary_sequence=args.primary_sequence, 
                                                                isotropic=True, 
                                                                msg=msg)
        
        # Save affine after resampling
        affine_after_resampling = mris[args.primary_sequence].affine
        header_after_resampling = mris[args.primary_sequence].header
        
        # 3. Crop and Concatenate
        mris_array, annotations_array, msg = crop_and_concatenate(
            mris, annotations, primary_sequence=args.primary_sequence, 
            save_sequence_order=args.save_sequence_order, msg=msg)
        
        # 4. Combine annotations
        annotations_array, msg = combine_annotations(annotations_array, None, msg)
        
        # Convert to Nifti1Image
        mris_image = nib.Nifti1Image(mris_array.astype(np.float32), affine_after_resampling, header_after_resampling)
        annotations_image = nib.Nifti1Image(annotations_array.astype(np.uint8), affine_after_resampling, header_after_resampling)
        
        # Check Annotations Stats
        msg += "\tChecking new stats for annotations after transforms\n"
        _, metadata, msg = process_cmb_mask(annotations_image, msg)
        annotations_metadata_new = {args.primary_sequence: metadata}


        # Save to Disk
        nib.save(mris_image, os.path.join(args.data_dir_path, subject, args.mris_subdir, subject + '.nii.gz'))
        nib.save(annotations_image, os.path.join(args.data_dir_path, subject, args.annotations_subdir, subject + '.nii.gz'))
        
        # Convert numpy arrays to lists
        labels_metadata_listed = numpy_to_list(labels_metadata)
        annotations_metadata_new_listed = numpy_to_list(annotations_metadata_new)

        # Save Metadata for CMBs using JSON format
        with open(os.path.join(args.data_dir_path, subject, args.annotations_metadata_subdir, f'{subject}_raw.json'), "w") as file:
            json.dump(labels_metadata_listed, file, indent=4)
        with open(os.path.join(args.data_dir_path, subject, args.annotations_metadata_subdir, f'{subject}_processed.json'), "w") as file:
            json.dump(annotations_metadata_new_listed, file, indent=4)
    
    except Exception:
            
        msg += f'Failed to process {subject}!\n\nException caught: {traceback.format_exc()}'
    
    # Finalize
    end = time.time()
    msg += f'Finished processing of {subject} in {end - start} seconds!\n\n'
    write_to_log_file(msg, args.log_file_path)


def main(args):

    args.dataset_dir_path = os.path.join(args.output_dir, args.dataset_name)
    args.data_dir_path = os.path.join(args.dataset_dir_path, 'Data')
    args.mris_subdir =  'MRIs'
    args.annotations_subdir =  'Annotations'
    args.annotations_metadata_subdir = 'Annotations_metadata'
    
    for dir_p in [args.output_dir, args.dataset_dir_path, args.data_dir_path]:
        ensure_directory_exists(dir_p)

    current_time = datetime.now()
    current_datetime = current_time.strftime("%d%m%Y_%H%M%S")
    args.log_file_path = os.path.join(args.dataset_dir_path, f'log_{current_datetime}.txt')

    # Ignore folder starting with weird symbol
    subjects = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    # Determine number of worker processes
    available_cpu_count = multiprocessing.cpu_count()
    num_workers = min(args.num_workers, available_cpu_count)
    
    # Parallelizing using multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(partial(process_study, args), subjects), total=len(subjects)))

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
                        help='Name of the dataset. If VALDO, include "VALDO" in the name')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    
