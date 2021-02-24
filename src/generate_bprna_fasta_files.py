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

brackets = ["<", ">", "[", "]", "{", "}", "A", "a", "B", "b", "C", "c", "D", "d", "E", "e"]

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  bprna_dir_path = asset_dir_path + "/bprna_data"
  fasta_file_path = asset_dir_path + "/bprna.fa"
  buf = ""
  seq_num = 0
  for bprna_file in os.listdir(bprna_dir_path):
    bprna_file_path = os.path.join(bprna_dir_path, bprna_file)
    with open(bprna_file_path) as f:
      lines = [line for line in f.readlines() if not line.startswith("#")]
      (seq, dbn) = (lines[0].strip(), remove_pseudoknots(lines[1].strip()))
      if not is_valid(seq):
        continue
      name = os.path.splitext(bprna_file)[0]
      buf += ">%s %s\n%s\n" % (name, dbn, seq)
      seq_num += 1
  with open(fasta_file_path, "w") as f:
    f.write(buf)
  cd_hit_est_output_file_path = asset_dir_path + "/bprna_cd_hit_est.fa"
  cmd = "cd-hit-est -c 0.8 -i " + fasta_file_path + " -o " + cd_hit_est_output_file_path
  utils.run_command(cmd)
  recs = [rec for rec in SeqIO.parse(cd_hit_est_output_file_path, "fasta")]
  print("%d of sequences are reduced to %d of sequences" % (seq_num, len(recs)))
  contrafold_train_data_dir_path = asset_dir_path + "/train_data_4_contrafold"
  if not os.path.isdir(contrafold_train_data_dir_path):
    os.mkdir(contrafold_train_data_dir_path)
  for rec in recs:
    splits = rec.description.split()
    (name, dbn) = (splits[0], splits[1])
    contrafold_train_datum_file_path = os.path.join(contrafold_train_data_dir_path, name + ".fa")
    with open(contrafold_train_datum_file_path, "w") as f:
      f.write(">seq\n%s\n>second_struct\n%s" % (str(rec.seq), dbn))

def remove_pseudoknots(dbn):
  result_str = dbn
  str_len = len(dbn)
  for i in range(str_len):
    char = dbn[i]
    if char in brackets:
      result_str = result_str[: i] + "." + result_str[i + 1 :]
  return result_str

def is_valid(seq):
  if any(char in str(seq) for char in "RYWSMKHBVDN"):
    return False
  else:
    return True

if __name__ == "__main__":
  main()
