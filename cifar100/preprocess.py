""" preprocess.py

Preprocessing for CIFAR-100 dataset.

This module provides functions to download and join the complete
CIFAR-100 dataset. The jointly dataset is saved at
${DATASETS}/cifar100.h5.

Example
-------
To run:
  python preprocess.py run

"""

import os
import pathlib
import pickle
from os.path import join

import fire
import numpy as np
import zarr
from dotenv import load_dotenv
from tqdm import tqdm

import common


load_dotenv()
DATASETS_DIR = os.getenv('DATASETS_DIR')
URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
FILENAME = 'cifar-100-python.tar.gz'
DS_DIR = join(DATASETS_DIR, 'cifar100')
IMAGES_DIR = join(DS_DIR, 'images')
DATA_DIR = join(DS_DIR, 'cifar-100-python')
MERGE_DIR = join(DS_DIR, 'cifar100.zarr')


def _load_cifar_set(filepath):
  """Load train/test subsets from ${DATASETS_DIR}/cifar100/"""
  with open(filepath, 'rb') as f:
    d = pickle.load(f, encoding='bytes')
    coarses = d[b'coarse_labels']
    fines = d[b'fine_labels']
    names = d[b'filenames']
    imgs = d[b'data']
    imgs = imgs / 255.0
    imgs = imgs.reshape((-1, 3, 32, 32))
    imgs = imgs.transpose((0, 2, 3, 1))
    imgs = np.vsplit(imgs, imgs.shape[0])
    imgs = list(np.squeeze(imgs))
    return coarses, fines, names, imgs


def download():
  """Downloads and extracts at ${DATASETS_DIR}/cifar100."""
  print('download() running ...')
  common.utils.download(URL, DS_DIR, FILENAME, extract='auto')
  print(f'Dataset extracted at {DATA_DIR}')


def merge():
  """Merge train/test subsets at ${DATASETS_DIR}/cifar100/cifar100.h5."""
  print('merge() running ...')
  trn = _load_cifar_set(os.path.join(DATA_DIR, 'train'))
  tst = _load_cifar_set(os.path.join(DATA_DIR, 'test'))
  src_data = [l1 + l2 for l1, l2 in zip(trn, tst)]

  data = {}
  for coarse, fine, name, img in zip(*src_data):
    coarse = f'{coarse:02d}'
    fine = f'{fine:02d}'
    name = name.decode(encoding='utf-8').split('.')[0]
    if coarse not in data:
      data[coarse] = {}
    if fine not in data[coarse]:
      data[coarse][fine] = {}
    data[coarse][fine][name] = img

  f = zarr.open(MERGE_DIR, 'w')
  for coarse in tqdm(sorted(data.keys())):
    coarse_grp = f.create_group(coarse)
    for fine in sorted(data[coarse].keys()):
      fine_grp = coarse_grp.create_group(fine)
      for name in sorted(data[coarse][fine].keys()):
        img = data[coarse][fine][name]
        fine_grp.create_dataset(name, data=img, dtype=np.float32)

  print(f'zarr dataset saved to {MERGE_DIR}')


def run():
  download()
  merge()


if __name__ == '__main__':
  fire.Fire()
