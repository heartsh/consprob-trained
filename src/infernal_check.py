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
  test_data_dir_path = asset_dir_path + "/test_data"
  test_ref_ss_dir_path = asset_dir_path + "/test_ref_sss"
  test_ref_sa_dir_path = asset_dir_path + "/test_ref_sas"
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  if not os.path.isdir(infernal_black_list_dir_path):
    os.mkdir(infernal_black_list_dir_path)
  temp_dir_path = "/tmp/infernal_check_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  if False:
    for sa_file in os.listdir(test_ref_sa_dir_path):
      if not sa_file.endswith(".sth"):
        continue
      (sa_file_name, extension) = os.path.splitext(sa_file)
      sa_file_path = os.path.join(test_ref_sa_dir_path, sa_file)
      infernal_output_file_path = os.path.join(test_ref_sa_dir_path, sa_file_name + "_infernal.dat")
      infernal_build_command = "cmbuild -F " + infernal_output_file_path + " " + sa_file_path
      utils.run_command(infernal_build_command)
      infernal_calib_command = "cmcalibrate " + infernal_output_file_path
      utils.run_command(infernal_calib_command)
  temp_seq_file_path = os.path.join(temp_dir_path, "temp.fa")
  temp_seq_file = open(temp_seq_file_path, "w")
  for seq_file in os.listdir(train_data_dir_path):
    if not seq_file.endswith(".fa"):
      continue
    seq_file_path = os.path.join(train_data_dir_path, seq_file)
    # temp_seq_file_path = os.path.join(temp_dir_path, seq_file)
    # temp_seq_file = open(temp_seq_file_path, "w")
    for j, rec in enumerate(SeqIO.parse(seq_file_path, "fasta")):
      if j >= 2:
        break
      seq_with_gaps = str(rec.seq)
      seq = seq_with_gaps.replace("-", "")
      temp_seq_file.write(">%s\n%s\n" % (rec.id, seq))
  temp_seq_file.close()
  for sa_file in os.listdir(test_ref_sa_dir_path):
    if not sa_file.endswith("_infernal.dat"):
      continue
    (sa_file_name, extension) = os.path.splitext(sa_file)
    sa_file_path = os.path.join(test_ref_sa_dir_path, sa_file)
    infernal_black_list_file_path = os.path.join(infernal_black_list_dir_path, sa_file)
    infernal_search_command = "cmsearch -E 0.001 " + sa_file_path + " " + temp_seq_file_path
    (output, _, _) = utils.run_command(infernal_search_command)
    # print(str(output))
    if "No hits detected" not in str(output):
      shutil.copyfile(sa_file_path, infernal_black_list_file_path)
      print(sa_file_path)

  shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
  main()
