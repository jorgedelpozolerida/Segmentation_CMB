#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Utility functions to load project related data


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
import nibabel as nib
import json

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Get the logger for the package
logger_nib = logging.getLogger('nibabel')

# Set the log level to CRITICAL to deactivate normal logging
logger_nib.setLevel(logging.CRITICAL)

def read_all_niftis_raw(sub, data_dir):
    sub_dir = f"{data_dir}/sub-{sub}"
    mask = nib.load(f"{sub_dir}/sub-{sub}_space-T2S_CMB.nii.gz")
    t2s = nib.load(f"{sub_dir}/sub-{sub}_space-T2S_desc-masked_T2S.nii.gz")
    t2 = nib.load(f"{sub_dir}/sub-{sub}_space-T2S_desc-masked_T2.nii.gz")
    t1 = nib.load(f"{sub_dir}/sub-{sub}_space-T2S_desc-masked_T1.nii.gz")

    return mask, t2s, t2, t1


def read_data_processed(sub, data_dir):
    sub_dir = f"{data_dir}/sub-{sub}"
    cmb, mri = nib.load(f"{sub_dir}/Annotations/sub-{sub}.nii.gz"), \
                    nib.load(f"{sub_dir}/MRIs/sub-{sub}.nii.gz")
    with open(f"{sub_dir}/Annotations_metadata/sub-{sub}_raw.json", "rb") as file:
        metadata_raw = json.load(file)
    with open(f"{sub_dir}/Annotations_metadata/sub-{sub}_processed.json", "rb") as file:
        metadata_processed = json.load(file)
        
    return cmb, mri, metadata_raw, metadata_processed