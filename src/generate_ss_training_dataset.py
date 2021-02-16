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

bracket_pairs = [("(", ")"), ("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"), ("E", "e"), ]

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  train_data_dir_path = asset_dir_path + "/train_data"
  train_data_dir_path_4_ss_train = asset_dir_path + "/train_data_4_ss_train"
  if not os.path.isdir(train_data_dir_path_4_ss_train):
    os.mkdir(train_data_dir_path_4_ss_train)
  for train_datum_file in os.listdir(train_data_dir_path):
    if not train_datum_file.endswith(".fa"):
      continue
    train_datum_file_path = os.path.join(train_data_dir_path, train_datum_file)
    train_datum_file_path_4_ss_train = os.path.join(train_data_dir_path_4_ss_train, train_datum_file)
    recs = [rec for rec in SeqIO.parse(train_datum_file_path, "fasta")]
    css = recs[-1].seq
    sampled_index = numpy.random.choice(range(0, len(recs) - 1), 1, replace = False)[0]
    sampled_seq = recs[int(sampled_index)].seq
    recovered_ss = recover_ss(css, sampled_seq)
    train_datum_file_4_ss_train = open(train_datum_file_path_4_ss_train, "w")
    train_datum_file_4_ss_train.write(">seq\n%s\n>second_struct\n%s\n" % (str(sampled_seq).replace("-", ""), recovered_ss))

def recover_ss(css, seq_with_gaps):
  pos_map = {}
  pos = 0
  for (i, char) in enumerate(seq_with_gaps):
    if char != "-":
      pos_map[i] = pos
      pos += 1
  recovered_ss = "." * pos
  stack = []
  for (left, right) in bracket_pairs:
    for (i, char) in enumerate(css):
      if char == left:
        stack.append(i)
      elif char == right:
        j = stack.pop()
        if seq_with_gaps[j] == "-" or seq_with_gaps[i] == "-":
          continue
        mapped_j = pos_map[j]
        mapped_i = pos_map[i]
        recovered_ss = recovered_ss[: mapped_j] + left + recovered_ss[mapped_j + 1 :]
        recovered_ss = recovered_ss[: mapped_i] + right + recovered_ss[mapped_i + 1 :]
  return recovered_ss

if __name__ == "__main__":
  main()
