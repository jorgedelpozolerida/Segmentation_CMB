#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Generates split file for dataset

Two different split types implemented, both keeping healthy-ill patients ratio
for each split.

@author: jorgedelpozolerida
@date: 29/10/2023
"""

import os
import sys
import argparse
import traceback


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import pickle

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

import json
import os
import random
from collections import defaultdict

def create_splits(data_dir, seed=42):
    """
    Splits data into train, validation, and optionally test sets while maintaining the proportion of healthy and unhealthy subjects.
    
    Args:
        data_dir (str): Directory containing the subfolders for each subject.
        split_type (str): Type of split ('train_val_test' or 'train_val').
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary with keys 'train', 'validation', 'test' (if applicable) and corresponding subfolder lists as values.
    """
    random.seed(seed)

    healthy_subjects = []
    unhealthy_subjects = []

    # Iterate over subfolders and classify subjects based on the number of CMBs
    for subj_folder in os.listdir(data_dir):
        subj_path = os.path.join(data_dir, subj_folder)
        if os.path.isdir(subj_path):
            metadata_filepath = os.path.join(subj_path, 'Annotations_metadata' , f'{subj_folder}_raw.json')

            if os.path.exists(metadata_filepath):
                with open(metadata_filepath, 'rb') as file:
                    metadata_dict = json.load(file)

                if len(metadata_dict['T2S']['centers_of_mass']) == 0:
                    healthy_subjects.append(subj_folder)
                else:
                    unhealthy_subjects.append(subj_folder)

    # Shuffle lists
    random.shuffle(healthy_subjects)
    random.shuffle(unhealthy_subjects)

    # Calculate split sizes
    num_healthy = len(healthy_subjects)
    num_unhealthy = len(unhealthy_subjects)

    healthy_train_size = int(num_healthy * 0.7)
    unhealthy_train_size = int(num_unhealthy * 0.7)

    # Create splits
    splits = defaultdict(list)
    splits['train'] = healthy_subjects[:healthy_train_size] + unhealthy_subjects[:unhealthy_train_size]

    if args.split_type == 'train_val':
        healthy_val_size = num_healthy - healthy_train_size
        unhealthy_val_size = num_unhealthy - unhealthy_train_size

        splits['valid'] = healthy_subjects[healthy_train_size:] + unhealthy_subjects[unhealthy_train_size:]

    elif args.split_type == 'train_val_test':
        healthy_val_test_size = num_healthy - healthy_train_size
        unhealthy_val_test_size = num_unhealthy - unhealthy_train_size

        splits['valid'] = healthy_subjects[healthy_train_size:healthy_train_size + healthy_val_test_size // 2] + unhealthy_subjects[unhealthy_train_size:unhealthy_train_size + unhealthy_val_test_size // 2]
        splits['test'] = healthy_subjects[healthy_train_size + healthy_val_test_size // 2:] + unhealthy_subjects[unhealthy_train_size + unhealthy_val_test_size // 2:]
    
    # Shuffle splits to mix healthy and unhealthy subjects
    for key in splits:
        random.shuffle(splits[key])

    return splits


def main(args):

    
    data_dir = os.path.join(args.dataset_dir, "Data")

    splits_dict = create_splits(data_dir, seed=42)

    # Save to splits.json
    split_path = os.path.join(data_dir, 'splits.json')
    _logger.info(f"File path: {split_path}")
    if os.path.exists(split_path) and not args.overwrite:
        _logger.warning("Splits already exist, add overwrite flag if wanted")
    else:
        with open(split_path, 'w') as f:
            json.dump(splits_dict, f)

        _logger.info("Splits created and saved to splits.json.")


    return


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Add this flag if you want to overwrite file')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to the dataset folder of dataset which has Data/ folder inside with subjects')
    parser.add_argument('--split_type', type=str, choices=['train_val_test', 'train_val'], 
                        default='train_val_test', help='Type of split to create (default: train_val_test)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)