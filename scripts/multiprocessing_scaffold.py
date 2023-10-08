#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for parallel doing stuff


SCAFFOLD


@author: jorgedelpozolerida
@date: 06/10/2023
"""



import os
import sys

import time  # NOQA E402
import logging  # NOQA E402
import argparse  # NOQA E402
import traceback  # NOQA E402
from tqdm import tqdm  # NOQA E402
import multiprocessing as mp  # NOQA E402
import pandas as pd

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def target_function(args, cases_pending, cases_done, processes_done):

    while cases_pending.qsize() > 0:
        dat1, dat2 = cases_pending.get() # NOTE: get data you need here
        
        # DO STUFF HERE

        cases_done.put(None) # NOTE: save data you need here

    processes_done.put(1)


def do_SOMETHING_on_all_cases(args):

    # NOTE: replace here with your actual data
    data1 = [1,2,3]
    data2 = ["a", "b", "c"]
    
    # initiate multiprocessing queues
    cases_pending = mp.Queue()
    cases_done = mp.Queue()
    processes_done = mp.Queue()

    # put all cases in queue with metadata
    for dat1, dat2 in zip(data1, data2):
        cases_pending.put(dat1, dat2) # NOTE: add here metadata to share to tasks

    # initialize progress bar
    if args.progress_bar:
        progress_bar = tqdm(total=cases_pending.qsize())
        number_of_cases_done_so_far = 0

    # start processes
    processes = []
    if cases_pending.qsize() > 0:
        number_of_workers = min(args.number_of_workers,
                                cases_pending.qsize())
        for i in range(number_of_workers):
            process = mp.Process(target=target_function, # NOTE: put your target function here 
                                 args=(args, cases_pending, cases_done, processes_done))
            processes.append(process)
            process.start()

        while True:
            if args.progress_bar:
                number_of_cases_done_now = cases_done.qsize()
                difference = number_of_cases_done_now - number_of_cases_done_so_far
                if difference > 0:
                    progress_bar.update(difference)
                    number_of_cases_done_so_far = number_of_cases_done_now

            if processes_done.qsize() == number_of_workers:
                if args.progress_bar:
                    progress_bar.close()
                for process in processes:
                    process.terminate()
                break

            time.sleep(0.1)


def main(args):

    try:
        do_SOMETHING_on_all_cases(args)

    except Exception:
        _logger.error('Exception caught in main: {}'.format(
            traceback.format_exc()))
        return 1
    return 0


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--number_of_workers', type=int, default=5,
                        help='Number of workers running in parallel')
    parser.add_argument('--progress_bar', type=bool, default=True,
                        help='Whether or not to show a progress bar')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)