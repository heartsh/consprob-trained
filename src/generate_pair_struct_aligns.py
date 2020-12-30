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
  rfam_seed_sta_file_path = asset_dir_path + "/rfam_seed_stas_v14.3.sth"
  train_data_dir_path = asset_dir_path + "/train_data"
  train_data_dir_path_4_micro_bench = asset_dir_path + "/train_data_4_micro_bench"
  test_data_dir_path = asset_dir_path + "/test_data"
  test_ref_sa_dir_path = asset_dir_path + "/test_ref_sas"
  test_ref_sa_dir_path_4_micro_bench = asset_dir_path + "/test_ref_sas_4_micro_bench"
  if not os.path.isdir(train_data_dir_path):
    os.mkdir(train_data_dir_path)
  if not os.path.isdir(train_data_dir_path_4_micro_bench):
    os.mkdir(train_data_dir_path_4_micro_bench)
  if not os.path.isdir(test_data_dir_path):
    os.mkdir(test_data_dir_path)
  if not os.path.isdir(test_ref_sa_dir_path):
    os.mkdir(test_ref_sa_dir_path)
  if not os.path.isdir(test_ref_sa_dir_path_4_micro_bench):
    os.mkdir(test_ref_sa_dir_path_4_micro_bench)
  max_sa_len = 500
  max_seq_num = 20
  stas = [sta for sta in AlignIO.parse(rfam_seed_sta_file_path, "stockholm") if len(sta[0]) <= max_sa_len and len(sta) <= max_seq_num and is_valid(sta)]
  num_of_stas = len(stas)
  print("# RNA families: %d" % num_of_stas)
  shuffle(stas)
  train_data_num = int(0.5 * num_of_stas)
  test_data_num = num_of_stas - train_data_num
  train_data, test_data = train_test_split(stas, test_size = 0.5);
  sample_rate = 0.02
  num_of_samples = int(sample_rate * train_data_num)
  print("# RNA families for micro benchmark: %d" % num_of_samples)
  sampled_train_data = numpy.random.choice(train_data, num_of_samples, replace = False)
  counter = 0
  for train_datum in train_data:
    cons_second_struct = convert_cons_second_struct(train_datum.column_annotations["secondary_structure"])
    for i, seq_1 in enumerate(train_datum):
      for seq_2 in train_datum[i + 1 :]:
        train_datum_file_path = os.path.join(train_data_dir_path, "train_datum_%d.fa" % counter)
        train_datum_file = open(train_datum_file_path, "w")
        train_datum_file.write(">seq_1\n%s\n\n>seq_2\n%s\n\n>cons_second_struct\n%s" % (seq_1.seq, seq_2.seq, cons_second_struct))
        counter += 1
  for i, test_datum in enumerate(test_data):
    sa_file_path = os.path.join(test_ref_sa_dir_path, "rna_fam_%d.sth" % i)
    AlignIO.write(test_datum, sa_file_path, "stockholm")
    sa_file_path = os.path.join(test_ref_sa_dir_path, "rna_fam_%d.fa" % i)
    AlignIO.write(test_datum, sa_file_path, "fasta")
    sa_file_path = os.path.join(test_ref_sa_dir_path, "rna_fam_%d.aln" % i)
    AlignIO.write(test_datum, sa_file_path, "clustal")
    test_datum_file_path = os.path.join(test_data_dir_path, "rna_fam_%d.fa" % i)
    test_datum_file = open(test_datum_file_path, "w")
    for j, rec in enumerate(test_datum):
      test_datum_file.write(">%d(%s)\n%s\n" % (j, rec.id, str(rec.seq).replace("-", "")))
  counter = 0
  for i, train_datum in enumerate(sampled_train_data):
    cons_second_struct = convert_cons_second_struct(train_datum.column_annotations["secondary_structure"])
    for i, seq_1 in enumerate(train_datum):
      for seq_2 in train_datum[i + 1 :]:
        train_datum_file_path = os.path.join(train_data_dir_path_4_micro_bench, "train_datum_%d.fa" % counter)
        train_datum_file = open(train_datum_file_path, "w")
        train_datum_file.write(">seq_1\n%s\n\n>seq_2\n%s\n\n>cons_second_struct\n%s" % (seq_1.seq, seq_2.seq, cons_second_struct))
        counter += 1

def is_valid(sta):
  for row in sta:
    if any(char in str(row.seq) for char in "RYWSMKHBVDN"):
      return False
  return True

def convert_cons_second_struct(cons_second_struct):
  new_cons_second_struct = cons_second_struct.replace("<", "(").replace("[", "(").replace("{", "(").replace(">", ")").replace("]", ")").replace("}", ")")
  for i in range(len(new_cons_second_struct)):
    char = new_cons_second_struct[i]
    if char == "(" or char == ")":
      continue
    new_cons_second_struct = new_cons_second_struct[: i] + "." + new_cons_second_struct[i + 1 :]
  return new_cons_second_struct

if __name__ == "__main__":
  main()
