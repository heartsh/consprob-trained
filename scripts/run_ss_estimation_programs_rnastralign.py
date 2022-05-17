#! /usr/bin/env python

import utils
from Bio import SeqIO
import numpy
import seaborn
from matplotlib import pyplot
import os
import multiprocessing
import time
import datetime
import shutil
from os import path
from Bio import AlignIO
import glob

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  temp_dir_path = "/tmp/run_ss_estimation_programs_consalign_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  sub_thread_num = 8 if num_of_threads <= 8 else 4
  raf_params = []
  locarna_params = []
  dafs_params = []
  sparse_params = []
  consalign_params = []
  consalign_params_disabled_alifold = []
  consalign_params_turner = []
  consalign_params_trained = []
  consalign_params_trained_transfer = []
  consalign_params_trained_random_init = []
  consalign_params_transferred_only = []
  turbofold_params = []
  raf_dir_path = asset_dir_path + "/raf_rnastralign"
  locarna_dir_path = asset_dir_path + "/locarna_rnastralign"
  dafs_dir_path = asset_dir_path + "/dafs_rnastralign"
  sparse_dir_path = asset_dir_path + "/sparse_rnastralign"
  turbofold_dir_path = asset_dir_path + "/turbofold_rnastralign"
  consalign_dir_path = asset_dir_path + "/consalign_rnastralign"
  consalign_dir_path_disabled_alifold = asset_dir_path + "/consalign_rnastralign_disabled_alifold"
  consalign_dir_path_turner = asset_dir_path + "/consalign_rnastralign_turner"
  consalign_dir_path_trained_transfer = asset_dir_path + "/consalign_rnastralign_trained_transfer"
  consalign_dir_path_trained_random_init = asset_dir_path + "/consalign_rnastralign_trained_random_init"
  consalign_dir_path_transferred_only = asset_dir_path + "/consalign_rnastralign_transferred_only"
  if not os.path.isdir(raf_dir_path):
    os.mkdir(raf_dir_path)
  if not os.path.isdir(locarna_dir_path):
    os.mkdir(locarna_dir_path)
  if not os.path.isdir(dafs_dir_path):
    os.mkdir(dafs_dir_path)
  if not os.path.isdir(sparse_dir_path):
    os.mkdir(sparse_dir_path)
  if not os.path.isdir(consalign_dir_path):
    os.mkdir(consalign_dir_path)
  if not os.path.isdir(consalign_dir_path_disabled_alifold):
    os.mkdir(consalign_dir_path_disabled_alifold)
  if not os.path.isdir(consalign_dir_path_turner):
    os.mkdir(consalign_dir_path_turner)
  if not os.path.isdir(consalign_dir_path_trained_transfer):
    os.mkdir(consalign_dir_path_trained_transfer)
  if not os.path.isdir(consalign_dir_path_trained_random_init):
    os.mkdir(consalign_dir_path_trained_random_init)
  if not os.path.isdir(consalign_dir_path_transferred_only):
    os.mkdir(consalign_dir_path_transferred_only)
  if not os.path.isdir(turbofold_dir_path):
    os.mkdir(turbofold_dir_path)
  rna_dir_path = asset_dir_path + "/RNAStrAlign_sampled"
  for rna_sub_dir in os.listdir(rna_dir_path):
    rna_sub_dir_path = os.path.join(rna_dir_path, rna_sub_dir)
    rna_file_path = os.path.join(rna_sub_dir_path, "sampled_seqs.fa")
    if not os.path.isfile(rna_file_path):
      continue
    raf_sub_dir_path = os.path.join(raf_dir_path, rna_sub_dir)
    locarna_sub_dir_path = os.path.join(locarna_dir_path, rna_sub_dir)
    dafs_sub_dir_path = os.path.join(dafs_dir_path, rna_sub_dir)
    sparse_sub_dir_path = os.path.join(sparse_dir_path, rna_sub_dir)
    turbofold_sub_dir_path = os.path.join(turbofold_dir_path, rna_sub_dir)
    consalign_sub_dir_path = os.path.join(consalign_dir_path, rna_sub_dir)
    consalign_sub_dir_path_disabled_alifold = os.path.join(consalign_dir_path_disabled_alifold, rna_sub_dir)
    consalign_sub_dir_path_turner = os.path.join(consalign_dir_path_turner, rna_sub_dir)
    consalign_sub_dir_path_trained_transfer = os.path.join(consalign_dir_path_trained_transfer, rna_sub_dir)
    consalign_sub_dir_path_trained_random_init = os.path.join(consalign_dir_path_trained_random_init, rna_sub_dir)
    consalign_sub_dir_path_transferred_only = os.path.join(consalign_dir_path_transferred_only, rna_sub_dir)
    if not os.path.isdir(raf_sub_dir_path):
      os.mkdir(raf_sub_dir_path)
    if not os.path.isdir(locarna_sub_dir_path):
      os.mkdir(locarna_sub_dir_path)
    if not os.path.isdir(dafs_sub_dir_path):
      os.mkdir(dafs_sub_dir_path)
    if not os.path.isdir(sparse_sub_dir_path):
      os.mkdir(sparse_sub_dir_path)
    if not os.path.isdir(consalign_sub_dir_path):
      os.mkdir(consalign_sub_dir_path)
    if not os.path.isdir(consalign_sub_dir_path_disabled_alifold):
      os.mkdir(consalign_sub_dir_path_disabled_alifold)
    if not os.path.isdir(consalign_sub_dir_path_turner):
      os.mkdir(consalign_sub_dir_path_turner)
    if not os.path.isdir(consalign_sub_dir_path_trained_transfer):
      os.mkdir(consalign_sub_dir_path_trained_transfer)
    if not os.path.isdir(consalign_sub_dir_path_trained_random_init):
      os.mkdir(consalign_sub_dir_path_trained_random_init)
    if not os.path.isdir(consalign_sub_dir_path_transferred_only):
      os.mkdir(consalign_sub_dir_path_transferred_only)
    if not os.path.isdir(turbofold_sub_dir_path):
      os.mkdir(turbofold_sub_dir_path)
    raf_output_file_path = os.path.join(raf_sub_dir_path, rna_sub_dir + ".sth")
    locarna_output_file_path = os.path.join(locarna_sub_dir_path, rna_sub_dir + ".sth")
    dafs_output_file_path = os.path.join(dafs_sub_dir_path, rna_sub_dir + ".sth")
    sparse_output_file_path = os.path.join(sparse_sub_dir_path, rna_sub_dir + ".sth")
    raf_params.insert(0, (rna_file_path, raf_output_file_path))
    locarna_params.insert(0, (rna_file_path, locarna_output_file_path, False))
    dafs_params.insert(0, (rna_file_path, dafs_output_file_path))
    sparse_params.insert(0, (rna_file_path, sparse_output_file_path, True))
    consalign_params.insert(0, (rna_file_path, consalign_sub_dir_path, sub_thread_num, "ensemble", "trained_transfer", False))
    consalign_params_disabled_alifold.insert(0, (rna_file_path, consalign_sub_dir_path_disabled_alifold, sub_thread_num, "ensemble", "trained_transfer", True))
    consalign_params_turner.insert(0, (rna_file_path, consalign_sub_dir_path_turner, sub_thread_num, "turner", "trained_transfer", True))
    consalign_params_trained_transfer.insert(0, (rna_file_path, consalign_sub_dir_path_trained_transfer, sub_thread_num, "trained", "trained_transfer", True))
    consalign_params_trained_random_init.insert(0, (rna_file_path, consalign_sub_dir_path_trained_random_init, sub_thread_num, "trained", "trained_random_init", True))
    consalign_params_transferred_only.insert(0, (rna_file_path, consalign_sub_dir_path_transferred_only, sub_thread_num, "trained", "transferred_only", True))
    turbofold_params.insert(0, (rna_file_path, turbofold_sub_dir_path))
  pool = multiprocessing.Pool(int(num_of_threads / sub_thread_num))
  pool.map(run_consalign, consalign_params)
  pool.map(run_consalign, consalign_params_disabled_alifold)
  pool.map(run_consalign, consalign_params_turner)
  pool.map(run_consalign, consalign_params_trained_transfer)
  pool.map(run_consalign, consalign_params_trained_random_init)
  pool.map(run_consalign, consalign_params_transferred_only)
  pool = multiprocessing.Pool(num_of_threads)
  pool.map(run_raf, raf_params)
  pool.map(run_locarna, locarna_params)
  pool.map(run_dafs, dafs_params)
  pool.map(run_locarna, sparse_params)
  pool.map(run_turbofold, turbofold_params)
  shutil.rmtree(temp_dir_path)

def run_raf(raf_params):
  (rna_file_path, raf_output_file_path) = raf_params
  raf_command = "raf predict " + rna_file_path
  (output, _, _) = utils.run_command(raf_command)
  raf_output_file = open(raf_output_file_path, "w+")
  raf_output_file.write(output.decode())
  raf_output_file.close()
  sta = AlignIO.read(raf_output_file_path, "fasta")
  recs = sta[:-1]
  new_sta = AlignIO.MultipleSeqAlignment(recs)
  new_sta.column_annotations["secondary_structure"] = str(sta[-1].seq)
  AlignIO.write(new_sta, raf_output_file_path, "stockholm")

def run_dafs(dafs_params):
  (rna_file_path, dafs_output_file_path) = dafs_params
  dafs_command = "dafs " + rna_file_path
  (output, _, _) = utils.run_command(dafs_command)
  dafs_output_file = open(dafs_output_file_path, "w+")
  dafs_output_file.write(output.decode())
  dafs_output_file.close()
  sta = AlignIO.read(dafs_output_file_path, "fasta")
  recs = sta[1:]
  new_sta = AlignIO.MultipleSeqAlignment(recs)
  new_sta.column_annotations["secondary_structure"] = str(sta[0].seq)
  AlignIO.write(new_sta, dafs_output_file_path, "stockholm")

def run_locarna(locarna_params):
  (rna_file_path, locarna_output_file_path, is_sparse) = locarna_params
  locarna_command = "mlocarna " + rna_file_path + " --keep-sequence-order --width=10000"
  if is_sparse:
    locarna_command += " --sparse"
  (output, _, _) = utils.run_command(locarna_command)
  lines = [line.strip() for (i, line) in enumerate(str(output).split("\\n")) if i > 7]
  locarna_output_file = open(locarna_output_file_path, "w+")
  locarna_output_buf = "# STOCKHOLM 1.0\n\n"
  for line in lines:
    if line.startswith("alifold "):
      locarna_output_buf += "#=GC SS_cons %s\n//" % line.split()[1]
      break
    else:
      locarna_output_buf += line + "\n"
  locarna_output_file.write(locarna_output_buf)
  locarna_output_file.close()

def run_consalign(consalign_params):
  (rna_file_path, consalign_output_dir_path, sub_thread_num, scoring_model, train_type, disables_alifold) = consalign_params
  consalign_command = "consalign %s-t " % ("-d " if disables_alifold else "") + str(sub_thread_num) + " -i " + rna_file_path + " -o " + consalign_output_dir_path + " -m " + scoring_model + " -u " + train_type
  utils.run_command(consalign_command)

def run_turbofold(turbofold_params):
  (rna_file_path, turbofold_output_dir_path) = turbofold_params
  turbofold_command = "python2 /usr/local/linear_turbofold/linearturbofold -i %s -o %s" % (rna_file_path, turbofold_output_dir_path)
  utils.run_command(turbofold_command)
  recs = [rna_seq.seq for rna_seq in SeqIO.parse(rna_file_path, "fasta")]
  num_of_recs = len(recs)
  lines = [""] * (2 * num_of_recs)
  for i in range(num_of_recs):
    lines[2 * i] = ">%d\n" % i
  for ss_file in os.listdir(turbofold_output_dir_path):
    if not ss_file.endswith(".db"):
      continue
    ss_file_path = os.path.join(turbofold_output_dir_path, ss_file)
    reader = open(ss_file_path, "r")
    ss = reader.readlines()[-1].strip()
    reader.close()
    i = int(ss_file[0]) - 1
    lines[2 * i + 1] = ss + "\n\n"
  ss_output_file_path = turbofold_output_dir_path + "/output.fa"
  ss_output_file = open(ss_output_file_path, "w")
  ss_output_file.writelines(lines)
  ss_output_file.close()

def read_ct_file(ct_file_path):
  ct_file = open(ct_file_path, "r")
  lines = ct_file.readlines()
  seq_len = int(lines[0].split()[0])
  ss_string = ["." for i in range(seq_len)]
  num_of_lines = len(lines)
  for line in lines[1 : num_of_lines]:
    if "ENERGY" in line:
      break
    substrings = line.split()
    index_1 = int(substrings[0])
    index_2 = int(substrings[4])
    if index_2 == 0 or index_1 >= index_2:
      continue
    ss_string[index_1 - 1] = "("
    ss_string[index_2 - 1] = ")"
  return "".join(ss_string)

def get_sss(sta):
  css_string = sta.column_annotations["secondary_structure"]
  sta_len = len(sta[0])
  num_of_rnas = len(sta)
  pos_map_sets = []
  sss = []
  for i in range(num_of_rnas):
    pos_map_sets.append([])
    pos = -1
    for j in range(sta_len):
      char = sta[i][j]
      if char != "-":
        pos += 1
      pos_map_sets[i].append(pos)
    sss.append("." * (pos + 1))
  stack = []
  for i, char in enumerate(css_string):
    if char == "(" or char == "<" or char == "[" or char == "{":
      stack.append(i)
    elif char == ")" or char == ">" or char == "]" or char == "}":
      col_pos = stack.pop()
      for j in range(num_of_rnas):
        base_pair_1 = (sta[j][col_pos], sta[j][i])
        if base_pair_1[0] == "-" or base_pair_1[1] == "-":
          continue
        pos_pair_1 = (pos_map_sets[j][col_pos], pos_map_sets[j][i])
        sss[j] = sss[j][:pos_pair_1[0]] + "(" + sss[j][pos_pair_1[0] + 1:]
        sss[j] = sss[j][:pos_pair_1[1]] + ")" + sss[j][pos_pair_1[1] + 1:]
  return sss

if __name__ == "__main__":
  main()
