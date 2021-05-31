#! /usr/bin/env python

import utils
from Bio import SeqIO
from Bio import AlignIO
import numpy
import seaborn
from matplotlib import pyplot
import os
import multiprocessing
import time
import datetime
import shutil
from Bio.Align import MultipleSeqAlignment
from sklearn.model_selection import train_test_split
from random import shuffle

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  train_data_dir_path = asset_dir_path + "/train_data_4_contrafold"
  train_data_file_paths = [os.path.join(train_data_dir_path, file) for file in os.listdir(train_data_dir_path) if file.endswith(".fa")]
  contrafold_command = "contrafold train "
  for train_data_file_path in train_data_file_paths:
    contrafold_command += " " + train_data_file_path
  utils.run_command(contrafold_command)

if __name__ == "__main__":
  main()
