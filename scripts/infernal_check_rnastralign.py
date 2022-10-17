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
  rna_dir_path = asset_dir_path + "/RNAStrAlign_sampled"
  train_data_dir_path = asset_dir_path + "/train_data"
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list_rnastralign"
  if not os.path.isdir(infernal_black_list_dir_path):
    os.mkdir(infernal_black_list_dir_path)
  temp_dir_path = "/tmp/infernal_check_rnastralign_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  temp_seq_file_path = os.path.join(temp_dir_path, "temp.fa")
  temp_seq_file = open(temp_seq_file_path, "w")
  for seq_file in os.listdir(train_data_dir_path):
    if not seq_file.endswith(".fa"):
      continue
    seq_file_path = os.path.join(train_data_dir_path, seq_file)
    for j, rec in enumerate(SeqIO.parse(seq_file_path, "fasta")):
      if j >= 2:
        break
      seq_with_gaps = str(rec.seq)
      seq = seq_with_gaps.replace("-", "")
      temp_seq_file.write(">%s\n%s\n" % (rec.id, seq))
  temp_seq_file.close()
  for rna_sub_dir in os.listdir(rna_dir_path):
    rna_sub_dir_path = os.path.join(rna_dir_path, rna_sub_dir)
    rna_file = "%s.aln" % rna_sub_dir
    rna_file_path = os.path.join(rna_sub_dir_path, rna_file)
    infernal_output_file_path = os.path.join(rna_sub_dir_path, "infernal.dat")
    infernal_build_command = "cmbuild --noss -F " + infernal_output_file_path + " " + rna_file_path
    utils.run_command(infernal_build_command)
    infernal_search_command = "cmsearch " + infernal_output_file_path + " " + temp_seq_file_path
    (output, _, _) = utils.run_command(infernal_search_command)
    if "No hits detected" not in str(output):
      infernal_black_list_sub_dir_path = os.path.join(infernal_black_list_dir_path, rna_sub_dir)
      if not os.path.isdir(infernal_black_list_sub_dir_path):
        os.mkdir(infernal_black_list_sub_dir_path)
      infernal_black_list_file_path = os.path.join(infernal_black_list_sub_dir_path, rna_file)
      shutil.copyfile(rna_file_path, infernal_black_list_file_path)
  shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
  main()
