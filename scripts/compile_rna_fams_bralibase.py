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
from Bio.Seq import Seq
from math import ceil

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  bralibase_dir_path = asset_dir_path + "/data-set1"
  bralibase_dir_path_compiled = bralibase_dir_path + "_compiled"
  if not os.path.isdir(bralibase_dir_path_compiled):
    os.mkdir(bralibase_dir_path_compiled)
  param_sets = []
  for rna_fam_dir in os.listdir(bralibase_dir_path):
    rna_fam_dir_path = os.path.join(bralibase_dir_path, rna_fam_dir)
    if not os.path.isdir(rna_fam_dir_path):
      continue
    rna_fam_dir_path_compiled = os.path.join(bralibase_dir_path_compiled, rna_fam_dir)
    if not os.path.isdir(rna_fam_dir_path_compiled):
      os.mkdir(rna_fam_dir_path_compiled)
    align_dir_path = os.path.join(rna_fam_dir_path, "structural")
    seq_dir_path = os.path.join(rna_fam_dir_path, "unaligned")
    align_dir_path_compiled = os.path.join(rna_fam_dir_path_compiled, "structural")
    seq_dir_path_compiled = os.path.join(rna_fam_dir_path_compiled, "unaligned")
    if not os.path.isdir(align_dir_path_compiled):
      os.mkdir(align_dir_path_compiled)
    if not os.path.isdir(seq_dir_path_compiled):
      os.mkdir(seq_dir_path_compiled)
    for rna_fam_file in os.listdir(align_dir_path):
      align_file_path = os.path.join(align_dir_path, rna_fam_file)
      align = AlignIO.read(align_file_path, "fasta")
      if not is_valid(align):
        continue
      seq_file_path = os.path.join(seq_dir_path, rna_fam_file)
      align_file_path_compiled = os.path.join(align_dir_path_compiled, rna_fam_file)
      seq_file_path_compiled = os.path.join(seq_dir_path_compiled, rna_fam_file)
      align = AlignIO.read(align_file_path, "fasta")
      for rec in align:
        rec.seq = rec.seq.transcribe()
      AlignIO.write(align, align_file_path_compiled, "fasta")
      recs = [rec for rec in SeqIO.parse(seq_file_path, "fasta")]
      for rec in recs:
        rec.seq = rec.seq.transcribe()
      SeqIO.write(recs, seq_file_path_compiled, "fasta")

def is_valid(sta):
  for row in sta:
    if any(char in str(row.seq) for char in "RYWSMKHBVDN"):
      return False
  return True
  
if __name__ == "__main__":
  main()
