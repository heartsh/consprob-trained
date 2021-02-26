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
  rfam_seed_sta_file_path = asset_dir_path + "/rfam_seed_stas_v14.4.sth"
  all_data_dir_path = asset_dir_path + "/all_data"
  if not os.path.isdir(all_data_dir_path):
    os.mkdir(all_data_dir_path)
  max_sa_len = 500
  max_seq_num = 50
  stas = [sta for sta in AlignIO.parse(rfam_seed_sta_file_path, "stockholm") if len(sta[0]) <= max_sa_len and len(sta) <= max_seq_num and is_valid(sta)]
  num_of_stas = len(stas)
  print("# RNA families: %d" % num_of_stas)
  for i, datum in enumerate(stas):
    align_len = len(datum[0])
    num_of_recs = len(datum)
    datum_file_path = os.path.join(all_data_dir_path, "rna_fam_%d.fa" % i)
    datum_file = open(datum_file_path, "w")
    for j, rec in enumerate(datum):
      seq_with_gaps = str(rec.seq)
      datum_file.write(">%d(%s)\n%s\n" % (j, rec.id, seq_with_gaps.replace("-", "")))

def get_struct_srcs(rfam_file_path):
  struct_srcs = []
  with open(rfam_file_path, "r") as f:
    for line in f.readlines():
      if line.startswith("#=GF SS "):
        struct_srcs.append("Predicted; " in line)
  return struct_srcs
          

def is_valid(sta):
  for row in sta:
    if any(char in str(row.seq) for char in "RYWSMKHBVDN"):
      return False
  return True

def convert_css_without_pseudoknots(css):
  converted_css = ""
  for char in css:
    if char == "(" or char == "<" or char == "[" or char == "{":
      converted_css += "("
    elif char == ")" or char == ">" or char == "]" or char == "}":
      converted_css += ")"
    else:
      converted_css += "."
  return converted_css

def convert_css(css):
  converted_css = ""
  for char in css:
    if char == "(" or char == "<" or char == "[" or char == "{":
      converted_css += "("
    elif char == ")" or char == ">" or char == "]" or char == "}":
      converted_css += ")"
    elif char == "A" or char == "B" or char == "C" or char == "D" or char == "E" or char == "a" or char == "b" or char == "c" or char == "d" or char == "e":
      converted_css += char
    else:
      converted_css += "."
  return converted_css

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
