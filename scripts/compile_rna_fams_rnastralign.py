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

bracket_pairs = [("(", ")"), ("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"), ("E", "e"), ]
sampled_seq_num = 20
max_sa_len = 500

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  rnastralign_dir_path = asset_dir_path + "/RNAStrAlign"
  rnastralign_dir_path_compiled = rnastralign_dir_path + "_sampled"
  if not os.path.isdir(rnastralign_dir_path_compiled):
    os.mkdir(rnastralign_dir_path_compiled)
  param_sets = []
  for rna_fam_dir in os.listdir(rnastralign_dir_path):
    param_sets.insert(0, (rnastralign_dir_path, rnastralign_dir_path_compiled, rna_fam_dir))
  pool = multiprocessing.Pool(num_of_threads)
  pool.map(compile_rna_fam, param_sets)

def compile_rna_fam(params):
  (rnastralign_dir_path, rnastralign_dir_path_compiled, rna_fam_dir) = params
  rna_fam_dir_path = os.path.join(rnastralign_dir_path, rna_fam_dir)
  align_file = rna_fam_dir + ".fasta"
  align_file_path = os.path.join(rna_fam_dir_path, align_file)
  align = AlignIO.read(align_file_path, "fasta")
  rna_files = [rna_file for rna_file in os.listdir(rna_fam_dir_path) if rna_file.endswith(".seq")]
  num_of_rna_files = len(rna_files)
  indexes = [i for i in range(num_of_rna_files)]
  repeat_num = ceil(num_of_rna_files / sampled_seq_num)
  for r in range(repeat_num):
    offset = r * sampled_seq_num 
    sampled_indexes = [offset + i for i in range(sampled_seq_num)]
    sampled_rna_files = [rna_file for (i, rna_file) in enumerate(rna_files) if i in sampled_indexes]
    rna_fam_dir_sample = "%s_%d" % (rna_fam_dir, r)
    output_rna_dir_path = os.path.join(rnastralign_dir_path_compiled, rna_fam_dir_sample)
    mapped_indexes = {}
    for (i, row) in enumerate(align):
      mapped_indexes[row.id] = i
    lines = []
    sampled_align = []
    for (i, sampled_rna_file) in enumerate(sampled_rna_files):
      sampled_rna_file_path = os.path.join(rna_fam_dir_path, sampled_rna_file)
      rna_name = sampled_rna_file[: -4]
      with open(sampled_rna_file_path, "r") as f:
        line = ">%d\n" % i
        lines.append(line)
        read_lines = f.readlines()
        seq = read_lines[-1].strip()[:-1]
        line = seq + "\n\n"
        lines.append(line)
      mapped_index = mapped_indexes[rna_name]
      row = align[mapped_index]
      row.id = str(i)
      sampled_align.append(row)
    align_len = len(sampled_align[0])
    for i in reversed(range(align_len)):
      is_col_valid = False
      for row in sampled_align:
        seq = list(str(row.seq))
        if seq[i] != "-":
          is_col_valid = True
      if not is_col_valid:
        for row in sampled_align:
          seq = list(str(row.seq))
          seq[i] = ""
          seq = "".join(seq)
          row.seq = Seq(seq)
    sampled_align = AlignIO.MultipleSeqAlignment(sampled_align)
    if not (is_valid(sampled_align) and len(sampled_align[0]) <= max_sa_len and len(sampled_align) >= 2):
      continue
    if not os.path.isdir(output_rna_dir_path):
      os.mkdir(output_rna_dir_path)
    for (i, sampled_rna_file) in enumerate(sampled_rna_files):
      sampled_rna_file_path = os.path.join(rna_fam_dir_path, sampled_rna_file)
      rna_name = sampled_rna_file[: -4]
      sampled_struct_file = rna_name + ".ct"
      sampled_struct_file_path = os.path.join(rna_fam_dir_path, sampled_struct_file)
      if not os.path.isfile(sampled_struct_file_path):
        sampled_struct_file = rna_name + "_p1.ct"
        sampled_struct_file_path = os.path.join(rna_fam_dir_path, sampled_struct_file)
      output_struct_file_path = os.path.join(output_rna_dir_path, "%d.ct" % i)
      shutil.copyfile(sampled_struct_file_path, output_struct_file_path)
    output_rna_file_path = os.path.join(output_rna_dir_path, "sampled_seqs.fa")
    with open(output_rna_file_path, "w") as f:
      f.writelines(lines)
    output_align_file = rna_fam_dir_sample + ".aln"
    output_align_file_path = os.path.join(output_rna_dir_path, output_align_file)
    AlignIO.write(sampled_align, output_align_file_path, "clustal")

def is_valid(sta):
  for row in sta:
    if any(char in str(row.seq) for char in "RYWSMKHBVDN"):
      return False
  return True
  
if __name__ == "__main__":
  main()
