#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to colect metadata about new dataset created from AWS


{Long Description of Script}


@author: jorgedelpozolerida
@date: 10/12/2023
"""
import os
import sys
import argparse
import traceback


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402

import boto3
from typing import Tuple, List, Union
from tqdm import tqdm
import re 
import copy
from typing import Any, Dict



sys.path.append("/home/cerebriu/data/DM/data-management")

import aws_s3.utils_s3 as utils_s3

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def save_reports_output(args, reports_output):
    '''
    Saves reports output.
    Args:
        args (dict): Parsed arguments coming from parse_args() function.
        reports_output (pd.DataFrame): Dataframe containing parsed reports.
    '''
    _logger.info('Saving reports output...')

    reports_output_dir_path = args.out_dir
    if not os.path.isdir(reports_output_dir_path):
        os.makedirs(reports_output_dir_path)

    reports_output.to_csv(os.path.join(
        reports_output_dir_path, 'Parsed_Reports.csv'),
        index=False)

    _logger.info('\tSuccessfully saved reports output: {}!'.format(reports_output_dir_path))


def process_reports(report_paths):
    '''
    Reads reports and parses report impressions.
    Args:
        args (dict): Parsed arguments coming from parse_args() function.
        reports ([str]): List of paths to reports.
    Returns:
        reports_output (pd.DataFrame): Dataframe containing parsed reports.
    '''
    _logger.info('Parsing reports...')

    reports_output = []

    for report_path in tqdm(report_paths, desc='Reports Processing Progress'):
        report_file_name = os.path.basename(os.path.abspath(report_path))
        study_uid = report_file_name[:report_file_name.find('.txt')]

        with open(report_path, 'r') as report_file:
            report = report_file.read().rstrip()

        report = re.sub(r'\n\s*\n', '\n', report)

        patterns = ['impression', 'finding', 'result', 'conclusion']

        for pattern in patterns:
            impressions_idx = report.lower().rfind(pattern)
            if impressions_idx != -1:
                impression = report[(impressions_idx + len(pattern)):]
                regex = r'[^ a-zA-Z0-9\-\+\.\,\;\!\?\[\]\(\)]'
                impression = re.sub(regex, ' ', impression)
                impression = ' '.join(impression.split())
                break

        if impressions_idx == -1:
            impression = None
            tqdm.write('\tFailed to locate impression for study: {}'.format(study_uid))

        reports_output.append({'StudyInstanceUID': study_uid, 'ParsedReport': report,
                                'ParsedImpressions': impression})

    _logger.info('\tSuccessfully parsed reports!')

    reports_output = pd.DataFrame(reports_output)

    return reports_output



def download_report(args: Dict[str, Any], s3: Any, dataset_name: str, study_uid: str) -> int:
    """
    Downloads a report file from an S3 bucket based on the given study UID and dataset name.

    Parameters:
    args (Dict[str, Any]): Command-line arguments.
    s3 (Any): AWS S3 client.
    dataset_name (str): The name of the dataset to look for in S3 bucket.
    study_uid (str): StudyInstanceUID
    """

    # Construct the potential S3 report path
    report_aws_key = f"{dataset_name}/Restructured_Data_Annotations_V1/{study_uid}/{study_uid}.txt"
    
    # Define the local filename to save the downloaded report
    filename_out = os.path.join(args.rawdata_dir, f"{study_uid}.txt")
    
    # Check if the object exists in the S3 bucket and download if it does
    try:
        s3.download_file(args.bucket, report_aws_key, filename_out)
        if args.verbose:
            _logger.info(f"Succesfully downloaded: {f'{study_uid}.txt'}")
    except:
        _logger.error(f"Could not find on AWS: {report_aws_key}")
        return 0
    return 1


def main(args):


    s3 = boto3.client('s3')
    
    studies = pd.read_csv(args.studiescsv_path)
    finalprocessing_df = pd.read_csv(args.finalprocessingfilecsv_path)
    
    assert "StudyInstanceUID" in studies.columns, "Please provide CSV with StudyInstanceUID column"

    if "Dataset" not in studies.columns:
        _logger.info(f"No 'Dataset' column provided, will obtain dataset querying s3 for every study")
        utils_s3.confirm_execution()
        s3 = boto3.client('s3')
        studies['Dataset'] = [utils_s3.find_s3_dataset(s3, args.bucket, x) for x in tqdm(studies['StudyInstanceUID'], desc="Finding dataset for input studies")]

    # Handle missing studies
    df_nodataset = studies[studies['Dataset'].isnull()]
    if df_nodataset.shape[0] > 0:
        _logger.warning(f"Could not find Dataset column for these studies: {df_nodataset['StudyInstanceUID'].to_list()}")
        
    else:
        _logger.info("Succesfully found Dataset column for all studies")

    df_clean = studies[~studies['StudyInstanceUID'].isin(df_nodataset['StudyInstanceUID'])]
    df_clean = df_clean.sort_values("Dataset")

    for dataset in df_clean['Dataset'].unique():
        print(dataset)
        
        
    if args.process_reports:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
            
        assert args.out_dir is not None
        reports_paths = glob.glob(os.path.join(args.rawdata_dir, '*.txt'))
        
        reports_output = process_reports(reports_paths)
        save_reports_output(args, reports_output)

    return 1

def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--studiescsv_path', type=str, default=None, required=False,
                        help='Full path to csv containing at least column StudyInstanceUID')
    parser.add_argument('--finalprocessingfilecsv_path', type=str, default=None, required=False,
                        help='Full path to csv containing at least column StudyInstanceUID')
    parser.add_argument('--out_dir', type=str, default=None, required=False,
                            help='Full path to folder where to store processed Reports if --process_reports added')
    parser.add_argument('--cache_folder', type=str, default=os.getcwd(), required=False,
                            help='Full path to folder where to store intermediate files generated. Default to current working directory')
    parser.add_argument('--bucket', type=str, default='cerebriu-data-management-bucket',
                        help='s3 bucket to look for studies')
    parser.add_argument('--verbose',  default=False, action='store_true',
                        help='Add this flag if you want to see intermediate outputs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)