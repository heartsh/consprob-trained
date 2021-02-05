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
  train_data_dir_path = asset_dir_path + "/train_data_4_ss_train"
  train_data_file_paths = [os.path.join(train_data_dir_path, file) for file in os.listdir(train_data_dir_path) if file.endswith(".fa")]
  contrafold_command = "contrafold train "
  for train_data_file_path in train_data_file_paths:
    contrafold_command += " " + train_data_file_path
  utils.run_command(contrafold_command)

def recover_ss(css, seq_with_gaps):
  pos_map = {}
  pos = 0
  for (i, char) in enumerate(seq_with_gaps):
    if char != "-":
      pos_map[i] = pos
      pos += 1
  recovered_ss = "." * pos
  stack = []
  for (i, char) in enumerate(css):
    if char == "(":
      stack.append(i)
    elif char == ")":
      j = stack.pop()
      if seq_with_gaps[j] == "-" or seq_with_gaps[i] == "-":
        continue
      mapped_j = pos_map[j]
      mapped_i = pos_map[i]
      recovered_ss = recovered_ss[: mapped_j] + "(" + recovered_ss[mapped_j + 1 :]
      recovered_ss = recovered_ss[: mapped_i] + ")" + recovered_ss[mapped_i + 1 :]
  return recovered_ss

if __name__ == "__main__":
  main()
