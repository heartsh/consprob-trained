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

bracket_pairs = [("(", ")"), ("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"), ("E", "e"), ]
sampled_seq_num = 10

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  rfam_seed_sta_file_path = asset_dir_path + "/rfam_seed_stas_v14.4.sth"
  train_data_dir_path = asset_dir_path + "/train_data"
  train_data_dir_path_multi = asset_dir_path + "/train_data_multi"
  test_data_dir_path = asset_dir_path + "/test_data"
  test_ref_ss_dir_path = asset_dir_path + "/test_ref_sss"
  test_ref_sa_dir_path = asset_dir_path + "/test_ref_sas"
  if not os.path.isdir(train_data_dir_path):
    os.mkdir(train_data_dir_path)
  if not os.path.isdir(train_data_dir_path_multi):
    os.mkdir(train_data_dir_path_multi)
  if not os.path.isdir(test_data_dir_path):
    os.mkdir(test_data_dir_path)
  if not os.path.isdir(test_ref_ss_dir_path):
    os.mkdir(test_ref_ss_dir_path)
  if not os.path.isdir(test_ref_sa_dir_path):
    os.mkdir(test_ref_sa_dir_path)
  max_sa_len = 500
  max_seq_num = 20
  stas = [sta for sta in AlignIO.parse(rfam_seed_sta_file_path, "stockholm") if len(sta[0]) <= max_sa_len and len(sta) <= max_seq_num and is_valid(sta)]
  struct_srcs = get_struct_srcs(rfam_seed_sta_file_path)
  for i, sta in enumerate(stas):
    sta.annotations["is_predicted"] = struct_srcs[i]
  num_of_stas = len(stas)
  print("# RNA families: %d" % num_of_stas)
  train_data, test_data = train_test_split(stas, test_size = 0.5)
  param_sets = zip(list(range(0, len(train_data))), train_data)
  param_sets = [(params[0], params[1], train_data_dir_path, train_data_dir_path_multi) for params in param_sets]
  num_of_threads = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_of_threads)
  pool.map(write_train_datum, param_sets)
  param_sets = zip(list(range(0, len(test_data))), test_data)
  param_sets = [(params[0], params[1], test_data_dir_path, test_ref_sa_dir_path, test_ref_ss_dir_path) for params in param_sets]
  pool.map(write_test_datum, param_sets)

def write_train_datum(params):
  (i, train_datum, train_data_dir_path, train_data_dir_path_multi) = params
  if train_datum.annotations["is_predicted"]:
    return
  train_datum_file_path_multi = os.path.join(train_data_dir_path_multi, "train_datum_%d.sth" % i)
  AlignIO.write(train_datum, train_datum_file_path_multi, "stockholm")
  cons_second_struct = convert_css_without_pseudoknots(train_datum.column_annotations["secondary_structure"])
  seq_num = len(train_datum)
  indexes = [j for j in range(0, seq_num)]
  sampled_indexes = indexes if seq_num <= sampled_seq_num else numpy.random.choice(indexes, sampled_seq_num, replace = False).tolist()
  recs = [train_datum[j] for j in sampled_indexes]
  sampled_index_pairs = [(idx_1, idx_2) for (j, idx_1) in enumerate(sampled_indexes) for idx_2 in sampled_indexes[j + 1:]]
  for (j, sampled_index_pair) in enumerate(sampled_index_pairs):
    seq_1 = train_datum[int(sampled_index_pair[0])].seq
    seq_2 = train_datum[int(sampled_index_pair[1])].seq
    seq_1, seq_2, new_css = remove_invalid_cols(seq_1, seq_2, cons_second_struct)
    train_datum_file_path = os.path.join(train_data_dir_path, "train_datum_%d_%d.fa" % (i, j))
    train_datum_file = open(train_datum_file_path, "w")
    train_datum_file.write(">seq_1\n%s\n\n>seq_2\n%s\n\n>cons_second_struct\n%s" % (seq_1, seq_2, new_css))

def remove_invalid_cols(seq_1, seq_2, cons_second_struct):
  new_seq_1 = list(seq_1)
  new_seq_2 = list(seq_2)
  new_cons_second_struct = list(cons_second_struct)
  stack = []
  for i, char in enumerate(cons_second_struct):
    if char == "(":
      stack.append(i)
    elif char == ")":
      seq_char_pair = (seq_1[i], seq_2[i])
      col_pos = stack.pop()
      seq_char_pair_2 = (seq_1[col_pos], seq_2[col_pos])
      if seq_char_pair[0] == "-" or seq_char_pair[1] == "-" or seq_char_pair_2[0] == "-" or seq_char_pair_2[1] == "-":
        new_cons_second_struct[col_pos] = "."
        new_cons_second_struct[i] = "."
  for i in range(len(cons_second_struct)):
    char_pair = (seq_1[i], seq_2[i])
    if char_pair[0] == "-" and char_pair[1] == "-":
      new_seq_1[i] = ""
      new_seq_2[i] = ""
      new_cons_second_struct[i] = ""
  new_seq_1 = "".join(new_seq_1)
  new_seq_2 = "".join(new_seq_2)
  new_cons_second_struct = "".join(new_cons_second_struct)
  return new_seq_1, new_seq_2, new_cons_second_struct

def write_test_datum(params):
  (i, test_datum, test_data_dir_path, test_ref_sa_dir_path, test_ref_ss_dir_path) = params
  align_len = len(test_datum)
  indexes = [j for j in range(0, align_len)]
  css = convert_css(test_datum.column_annotations["secondary_structure"])
  sa_file_path = os.path.join(test_ref_sa_dir_path, "rna_fam_%d.sth" % i)
  AlignIO.write(test_datum, sa_file_path, "stockholm")
  test_datum_file_path = os.path.join(test_data_dir_path, "rna_fam_%d.fa" % i)
  ref_ss_file_path = os.path.join(test_ref_ss_dir_path, "rna_fam_%d.fa" % i)
  test_datum_file = open(test_datum_file_path, "w")
  ref_ss_file = open(ref_ss_file_path, "w")
  for j, rec in enumerate(test_datum):
    seq_with_gaps = str(rec.seq)
    test_datum_file.write(">%d(%s)\n%s\n" % (j, rec.id, seq_with_gaps.replace("-", "")))
    recovered_ss = recover_ss(css, seq_with_gaps)
    ref_ss_file.write(">%d(%s)\n%s\n" % (j, rec.id, recovered_ss))

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
