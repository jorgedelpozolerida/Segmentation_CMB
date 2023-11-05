#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Quick scirpt to get metadata about available CMB


{Long Description of Script}


@author: jorgedelpozolerida
@date: 05/11/2023
"""

import os
import sys
import argparse
import traceback


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


import os
import pandas as pd
import json

import os
import json
import pandas as pd



def extract_data(metadata, subdir):
    t2s_data = metadata.get('T2S', {})
    return {
        'subject_id': subdir,
        'centers_of_mass': t2s_data.get('centers_of_mass', []),
        'pixel_counts': t2s_data.get('pixel_counts', [])
    }

def expand_data(df):
    expanded_data = []
    
    for index, row in df.iterrows():
        # Pair each center of mass with the corresponding pixel count
        expanded_data.extend(
            {
                'subject_id': row['subject_id'],
                'n_CMB': len(row['centers_of_mass']), 
                'center_of_mass': f"{int(center[0])}, {int(center[1])}, {int(center[2])}",
                'n_voxels': count,
            }
            for center, count in zip(
                row['centers_of_mass'], row['pixel_counts']
            )
        )
    return pd.DataFrame(expanded_data)

def main(args):
    root_dir = '/home/cerebriu/data/datasets/VALDO_processed/VALDO_processed/Data/'
    data_processed = []
    data_raw = []

    # Loop through each subdirectory in the root directory
    for subdir in os.listdir(root_dir):
        # Processed JSON
        json_processed_path = os.path.join(root_dir, subdir, 'Annotations_metadata', f'{subdir}_processed.json')
        # Raw JSON
        json_raw_path = os.path.join(root_dir, subdir, 'Annotations_metadata', f'{subdir}_raw.json')

        # Check if the processed JSON file exists
        if os.path.isfile(json_processed_path):
            with open(json_processed_path, 'r') as file:
                metadata = json.load(file)
                data_processed.append(extract_data(metadata, subdir))

        # Check if the raw JSON file exists
        if os.path.isfile(json_raw_path):
            with open(json_raw_path, 'r') as file:
                metadata = json.load(file)
                data_raw.append(extract_data(metadata, subdir))
    
    # Convert the list of dictionaries to pandas DataFrames
    df_processed = pd.DataFrame(data_processed)
    df_raw = pd.DataFrame(data_raw)

    # Apply expansion to both DataFrames
    df_processed = expand_data(df_processed)
    df_raw = expand_data(df_raw)

    df_processed.sort_values(by=["subject_id", "n_CMB", "n_voxels"], inplace=True)
    df_raw.sort_values(by=["subject_id", "n_CMB", "n_voxels"], inplace=True)
    
    # Save the DataFrames to CSV files
    df_processed.to_csv("../data/CMB_processed.csv", index=False)
    df_raw.to_csv("../data/CMB_raw.csv", index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)