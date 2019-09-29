""" preprocess.py

Preprocessing for CIFAR-100 dataset.

This module provides a function to download and extract the CIFAR-100 dataset.

Example
-------
To run:
  python preprocess.py download

"""

import os
import pathlib
from os.path import join

import fire
from dotenv import load_dotenv

import common

load_dotenv()
DATASETS_DIR = os.getenv('DATASETS_DIR')
URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
DS_DIR = join(DATASETS_DIR, 'cifar100')
FILENAME = 'cifar-100-python.tar.gz'

def download():
  """Downloads and extracts the CIFAR-100 dataset.
  
  Dataset are extracted at ${DATASETS_DIR}/cifar100.
  """
  print('download() running ...')
  common.utils.download(URL, DS_DIR, FILENAME, extract='auto')

def run():
  download()

if __name__ == '__main__':
  fire.Fire()
