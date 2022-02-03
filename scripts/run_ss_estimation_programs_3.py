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

gammas = [2. ** i for i in range(-7, 11)]

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  temp_dir_path = "/tmp/run_ss_estimation_programs_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  raf_params = []
  locarna_params = []
  dafs_params = []
  sparse_params = []
  consalign_params = []
  turbofold_params = []
  turbofold_params_4_running_time = []
  raf_dir_path = asset_dir_path + "/raf"
  locarna_dir_path = asset_dir_path + "/locarna"
  dafs_dir_path = asset_dir_path + "/dafs"
  sparse_dir_path = asset_dir_path + "/sparse"
  turbofold_dir_path = asset_dir_path + "/turbofold"
  consalign_dir_path = asset_dir_path + "/consalign"
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
  if not os.path.isdir(turbofold_dir_path):
    os.mkdir(turbofold_dir_path)
  rna_dir_path = asset_dir_path + "/test_data"
  sub_thread_num = 4
  for rna_file in os.listdir(rna_dir_path):
    if not rna_file.endswith(".fa"):
      continue
    rna_file_path = os.path.join(rna_dir_path, rna_file)
    (rna_family_name, extension) = os.path.splitext(rna_file)
    raf_output_file_path = os.path.join(raf_dir_path, rna_family_name + ".fa")
    locarna_output_file_path = os.path.join(locarna_dir_path, rna_family_name + ".fa")
    dafs_output_file_path = os.path.join(dafs_dir_path, rna_family_name + ".fa")
    sparse_output_file_path = os.path.join(sparse_dir_path, rna_family_name + ".fa")
    consalign_output_dir_path = os.path.join(consalign_dir_path, rna_family_name)
    turbofold_output_dir_path = os.path.join(turbofold_dir_path, rna_family_name)
    if not os.path.isdir(turbofold_output_dir_path):
      os.mkdir(turbofold_output_dir_path)
    raf_params.insert(0, (rna_file_path, raf_output_file_path))
    locarna_params.insert(0, (rna_file_path, locarna_output_file_path, False))
    dafs_params.insert(0, (rna_file_path, dafs_output_file_path))
    sparse_params.insert(0, (rna_file_path, sparse_output_file_path, True))
    consalign_params.insert(0, (rna_file_path, consalign_output_dir_path))
    for gamma in gammas:
      gamma_str = str(gamma) if gamma < 1 else str(int(gamma))
      output_file = "gamma=" + gamma_str + ".fa"
      turbofold_output_file_path = os.path.join(turbofold_output_dir_path, output_file)
      turbofold_params.insert(0, (rna_file_path, turbofold_output_file_path, gamma, temp_dir_path, rna_family_name))
      if gamma == 1.:
        turbofold_params_4_running_time.insert(0, (rna_file_path, turbofold_output_file_path, gamma, temp_dir_path, rna_family_name))
  pool = multiprocessing.Pool(int(num_of_threads / sub_thread_num))
  begin = time.time()
  pool.map(run_consalign, consalign_params)
  consalign_elapsed_time = time.time() - begin
  pool = multiprocessing.Pool(num_of_threads)
  if False:
    begin = time.time()
    pool.map(run_raf, raf_params)
    raf_elapsed_time = time.time() - begin
    begin = time.time()
    pool.map(run_locarna, locarna_params)
    locarna_elapsed_time = time.time() - begin
    begin = time.time()
    pool.map(run_dafs, dafs_params)
    dafs_elapsed_time = time.time() - begin
    begin = time.time()
    pool.map(run_locarna, sparse_params)
    sparse_elapsed_time = time.time() - begin
  pool.map(run_turbofold, turbofold_params)
  if False:
    begin = time.time()
    pool.map(run_turbofold, turbofold_params_4_running_time)
    turbofold_elapsed_time = time.time() - begin
    print("The elapsed time of RAF = %f [s]." % raf_elapsed_time)
    print("The elapsed time of LocARNA = %f [s]." % locarna_elapsed_time)
    print("The elapsed time of DAFS = %f [s]." % dafs_elapsed_time)
    print("The elapsed time of SPARSE = %f [s]." % sparse_elapsed_time)
    print("The elapsed time of ConsAlign = %f [s]." % consalign_elapsed_time)
    print("The elapsed time of TurboFold = %f [s]." % turbofold_elapsed_time)
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
  sss = get_sss(new_sta)
  buf = ""
  raf_output_file = open(raf_output_file_path, "w+")
  for (i, ss) in enumerate(sss):
    buf += ">%d\n%s\n\n" % (i, ss)
  raf_output_file.write(buf)
  raf_output_file.close()

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
  sss = get_sss(new_sta)
  buf = ""
  dafs_output_file = open(dafs_output_file_path, "w+")
  for (i, ss) in enumerate(sss):
    buf += ">%d\n%s\n\n" % (i, ss)
  dafs_output_file.write(buf)
  dafs_output_file.close()

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
  sta = AlignIO.read(locarna_output_file_path, "stockholm")
  sss = get_sss(sta)
  buf = ""
  locarna_output_file = open(locarna_output_file_path, "w+")
  for (i, ss) in enumerate(sss):
    buf += ">%d\n%s\n\n" % (i, ss)
  locarna_output_file.write(buf)
  locarna_output_file.close()

def run_consalign(consalign_params):
  (rna_file_path, consalign_output_dir_path) = consalign_params
  consalign_command = "consalign -i " + rna_file_path + " -o " + consalign_output_dir_path
  utils.run_command(consalign_command)
  for consalign_output_file in glob.glob("consalign_*.sth"):
    consalign_output_file_path = os.path.join(consalign_output_dir_path, consalign_output_file)
    sta = AlignIO.read(consalign_output_file_path, "stockholm")
    sss = get_sss(sta)
    buf = ""
    consalign_output_file = open(consalign_output_file_path, "w+")
    for (i, ss) in enumerate(sss):
      buf += ">%d\n%s\n\n" % (i, ss)
    consalign_output_file.write(buf)
    consalign_output_file.close()

def run_turbofold(turbofold_params):
  (rna_file_path, turbofold_output_file_path, gamma, temp_dir_path, rna_family_name) = turbofold_params
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  rec_lens = [len(rec) for rec in recs]
  rec_seq_len = len(recs)
  turbofold_temp_dir_path = "%s/%s_gamma=%d" % (temp_dir_path, rna_family_name, gamma)
  if not os.path.isdir(turbofold_temp_dir_path):
    os.mkdir(turbofold_temp_dir_path)
  turbofold_config_file_contents = "InSeq = {"
  for i in range(rec_seq_len):
    turbofold_config_file_contents += "%s/%d.fasta;" % (turbofold_temp_dir_path, i)
  turbofold_config_file_contents += "}\nOutCT = {"
  for i in range(rec_seq_len):
    turbofold_config_file_contents += "%s/%d.ct;" % (turbofold_temp_dir_path, i)
  turbofold_config_file_contents += "}\nIterations = 3\nMode = MEA\nMeaGamma = %f" % gamma
  turbofold_config_file_path = os.path.join(turbofold_temp_dir_path, "turbofold_config.dat")
  turbofold_config_file = open(turbofold_config_file_path, "w")
  turbofold_config_file.write(turbofold_config_file_contents)
  turbofold_config_file.close()
  for (i, rec) in enumerate(recs):
    SeqIO.write([rec], open(os.path.join(turbofold_temp_dir_path, "%d.fasta" % i), "w"), "fasta")
  turbofold_command = "TurboFold " + turbofold_config_file_path
  utils.run_command(turbofold_command)
  turbofold_output_file_contents = ""
  all_files_exist = True
  for i in range(rec_seq_len):
    ct_file_path = os.path.join(turbofold_temp_dir_path, "%d.ct" % i)
    if path.exists(ct_file_path):
      ss_string = read_ct_file(ct_file_path)
      turbofold_output_file_contents += ">%d\n%s\n\n" % (i, ss_string)
    else:
      all_files_exist = False
      turbofold_output_file_contents = ""
      break
  if not all_files_exist:
    print("Some output files are empty. TurboFold is retried with # iterations = 1.")
    turbofold_config_file_contents = "InSeq = {"
    for i in range(rec_seq_len):
      turbofold_config_file_contents += "%s/%d.fasta;" % (turbofold_temp_dir_path, i)
    turbofold_config_file_contents += "}\nOutCT = {"
    for i in range(rec_seq_len):
      turbofold_config_file_contents += "%s/%d.ct;" % (turbofold_temp_dir_path, i)
    turbofold_config_file_contents += "}\nIterations = 1\nMode = MEA\nMeaGamma = %f" % gamma
    turbofold_config_file_path = os.path.join(turbofold_temp_dir_path, "turbofold_config.dat")
    turbofold_config_file = open(turbofold_config_file_path, "w")
    turbofold_config_file.write(turbofold_config_file_contents)
    turbofold_config_file.close()
    utils.run_command(turbofold_command)
    for i in range(rec_seq_len):
      ct_file_path = os.path.join(turbofold_temp_dir_path, "%d.ct" % i)
      ss_string = read_ct_file(ct_file_path)
      turbofold_output_file_contents += ">%d\n%s\n\n" % (i, ss_string)
  turbofold_output_file = open(turbofold_output_file_path, "w")
  turbofold_output_file.write(turbofold_output_file_contents)
  turbofold_output_file.close()

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
