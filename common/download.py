""" download.py

Download utils.
""" 


import math
import pathlib
import os

import requests
from tqdm import tqdm


def download(url, dst_dir, filename, proto='http', extract=None):
  """Downloads a file from a server.
 
  Parameters
  ----------
  url : str
    File URL to download.
  dst_dir : str
    Destination directory, created recursively if necessary.
  filename : str
    Final filename.
  proto : str
    Supported protocols: 'http'.
  extract : str
    Archive format to try for extracting the file. Options: 'tar', 'zip' or 'rar'.
    If 'rar', p7zip rar non-free module must be installed on the system. 
    Default None, file is not extracted.

  Raises
  ------
  ValueError
    If either `proto` or `extract` is unknown.

  """
  os.makedirs(dst_dir, exist_ok=True)
  filepath = os.path.join(dst_dir, filename)
  if os.path.exists(filepath):
    print(f"File {filepath} already exists, skiping download.")
    return
  print(f'Downloading {url} to {filepath}')
  if proto == 'http':
    _download_http(url, filepath)
  else:
    raise ValueError(f"Unsupported protocol {proto}")
  if extract:
    _extract(filepath, extract)
  


def _download_http(url, filepath):
  """Downloads a file from a HTTP server.
 
  Parameters
  ----------
  url : str
    File URL to download.
  filepath : str
    Destination filepath.

  """
  filepath = pathlib.Path(filepath)
  r = requests.get(url, stream=True)
  total_size = int(r.headers.get('content-length', 0))
  block_size = 1024
  with filepath.open('wb') as f:
    with tqdm(total=total_size, unit='iB', unit_scale=True) as t:
      for data in r.iter_content(block_size):
        t.update(len(data))
        f.write(data)
  if total_size != 0 and t.n != total_size:
    print(f"Error, expected {total_size} and downloaded {t.n} file sizes do not match.")


def _extract(filepath, format):
  """Extracts a file at the same directory using given format.
 
  Parameters
  ----------
  filepath : str
    Archive to extract.
  format : str
    Archive format to try for extracting the file.
    Options: 'auto', 'rar', 'tar', 'tar.gz', 'zip'.
    If 'auto', it will determine the format using the filename extention.
    If 'rar', p7zip rar non-free module should be installed on the system. 
    Default None, file is not extracted.

  Raises
  ------
  ValueError
    If `format` is unknown.

  """
  dst_dir = '/'.join(filepath.split('/')[:-1])
  if format == 'auto':
    format = '.'.join(filepath.split('/')[-1].split('.')[1:])
  print(f"Extracting using {format} format {filepath}")
  if format == 'rar':
    import subprocess
    cmd = f'7z x {filepath} -o{dst_dir}'
    try:
      subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as e:
      print(f"Error extracting rar: {filepath} {e.returncode} {e.output}")
  elif format == 'tar' or format == 'tar.gz':
    import tarfile
    with tarfile.open(filepath) as tf:
      tf.extractall(dst_dir)
  elif format == 'zip':
    import zipfile
    with zipfile.ZipFile(filepath) as z:
      z.extractall(dst_dir)
  else:
    raise ValueError(f"Unsupported format {format}")
