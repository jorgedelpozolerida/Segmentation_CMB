import os
import sys
import argparse
import traceback
import logging
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Define input and output paths
input_path = "/home/cerebriu/data/RESEARCH/test.txt"
output_path = "/home/cerebriu/data/RESEARCH/test.csv"

def process_file(input_path, output_path):
    """
    Process the input file and save results to the output file.
    """
    with open(input_path, encoding='utf-8') as input_file, \
            open(output_path, 'w', encoding='utf-8', newline='') as output_file:

        writer = csv.writer(output_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        # Write the header
        writer.writerow(["AD", "CADASIL", "SMART", "TBI", "ICH", "CAA", "CVA", "GT", "HD", "AS", "CMB-only"])

        row_id = 0
        values = []

        for line in tqdm(input_file, desc="Filtering and writing"):
            if row_id == 0:
                template = line.strip()
                print("Template:", template)

            row = line.strip()

            # Write values every 11 lines
            if row_id % 11 == 0 and row_id != 0:
                writer.writerow(values)
                values = []

            # Check if current row matches the template
            if row == template:
                val = template
            else:
                val = "yes"
            values.append(val)
            row_id += 1

        # Writing the last batch of values if any
        if values:
            writer.writerow(values)

if __name__ == "__main__":
    try:
        process_file(input_path, output_path)
    except Exception as e:
        _logger.error(f"An error occurred: {e}")
        traceback.print_exc()