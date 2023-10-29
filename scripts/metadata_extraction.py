#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to extract metadata from  "VALDO challenge (Task2)" dataset

Generates csv with all metadata extracted.

Note:
- Works in parallel using as many CPUs as specified

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
import shutil
from tqdm import tqdm
import csv
import nibabel as nib
import re
import multiprocessing

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
import re

# Constant to define order in which sequences are stacked together
MRI_ORDER = {
    "T2S": 0,
    "T2": 1,
    "T1": 2
}


def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_image_metadata(nifti_path):
    img = nib.load(nifti_path)
    xd, yd, zd = img.shape
    dx, dy, dz = img.header.get_zooms()
    data = img.get_fdata()
    axcodes = nib.aff2axcodes(img.affine)
    
    return {
        'shape': (xd, yd, zd),
        'zooms': (dx, dy, dz),
        'data_type': img.get_data_dtype(),
        'orientation': axcodes,
        'mean_pixel': np.nanmean(data),
        'min_pixel': np.nanmin(data),
        'max_pixel': np.nanmax(data),
        'data': data
    }

def process_study(args, subject):
    old_subject_dir = os.path.join(args.in_dir, subject)

    niftis = [n for n in os.listdir(old_subject_dir) if not n.startswith('._')]

    data = {
        'subject': [],  'X_dim': [], 'Y_dim': [], 
        'Z_dim': [], 'dx': [], 'dy': [], 'dz': [], 
        'has_nan': [], 'nan_percent': [],
        'pix_mean_val': [], 'pix_min_val': [], 'pix_man_val': [], 'Space': [],
        'Description': [], 
        'MRSequence': [], 'CMB_label': [], 'CMB_npix': [], 'data_type': [], 
        'orientation': [], 'filename': [], 'full_path': []
    }
    mri = {}

    for nifti in niftis:
        match = re.search(r'sub-\w+_space-(\w+)_desc-(\w+)_(\w+).', nifti)
        full_path = os.path.join(old_subject_dir, nifti)
        metadata = get_image_metadata(full_path)

        data['subject'].append(subject)
        data['filename'].append(nifti)
        data['X_dim'].append(metadata['shape'][0])
        data['Y_dim'].append(metadata['shape'][1])
        data['Z_dim'].append(metadata['shape'][2])
        data['dx'].append(metadata['zooms'][0])
        data['dy'].append(metadata['zooms'][1])
        data['dz'].append(metadata['zooms'][2])
        data['pix_mean_val'].append(metadata['mean_pixel'])
        data['pix_min_val'].append(metadata['min_pixel'])
        data['pix_man_val'].append(metadata['max_pixel'])
        data['data_type'].append(metadata['data_type'])
        data['orientation'].append(metadata['orientation'])
        data['full_path'].append(full_path)
        data['has_nan'].append(np.any(np.isnan(metadata['data'])) )
        data['nan_percent'].append(np.sum(np.isnan(metadata['data']))/len(metadata['data'].flatten())*100)


        if match:
            data['Space'].append(match[1])
            data['Description'].append(match[2])
            data['MRSequence'].append(match[3])
        else:
            data['Space'].append(None)
            data['Description'].append(None)
            data['MRSequence'].append("Label")

        if "CMB" in nifti:
            unique_labels = np.unique(metadata['data'])
            amount_each = [np.sum(metadata['data'] == l) for l in unique_labels]
            cmb_label = unique_labels[np.argmin(amount_each)]
            data['CMB_label'].append(np.argmin(amount_each) if len(unique_labels) == 2 else None)
            data['CMB_npix'].append(np.sum(metadata['data'] == cmb_label) if len(unique_labels) == 2 else None)
        else:
            data['CMB_label'].append(None)
            data['CMB_npix'].append(None)
            if match:
                mri[match[3]] = full_path

    assert len(mri) == 3, f"Following study has some issue: {subject}"

    return pd.DataFrame(data)


    
def worker(args_subject):
    '''
    Worker function for parallel processing of subjects.
    '''
    args, subject = args_subject
    try:
        return process_study(args, subject)
    except Exception as e:
        traceback.print_exc()
        _logger.error(f"Error processing subject {subject}: {e}")
        return None




def main(args):

    # Ignore folder starting with weird symbol
    subjects = [d for d in os.listdir(args.in_dir) if os.path.isdir(os.path.join(args.in_dir, d))]

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(args.num_workers)

    # Use the pool to process the subjects in parallel
    all_dataframes = pool.map(worker, [(args, subject) for subject in subjects])
    pool.close()
    pool.join()

    # Filter out any None values from failed processes
    all_dataframes = [df for df in all_dataframes if df is not None]

    # Concatenate all dataframes and save to CSV
    df_global = pd.concat(all_dataframes, ignore_index=True)
    df_global.to_csv(os.path.join(args.out_dir, f'{args.dataset_name}_metadata.csv'), index=False)

def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default=None, required=True,
                        help='Name of dataset being processed')
    parser.add_argument('--in_dir', type=str, default=None, required=True,
                        help='Path to the input directory of dataset')
    parser.add_argument('--out_dir', type=str, default=None, required=True,
                        help='Path to the output directory to save dataset')
    parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of workers running in parallel')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    