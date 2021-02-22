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

gammas = [2. ** i for i in range(-7, 11)]

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  temp_dir_path = "/tmp/run_ss_estimation_programs_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  ipknot_params = []
  knotty_params = []
  spot_rna_params = []
  raf_params = []
  locarna_params = []
  turbofold_params = []
  ipknot_dir_path = asset_dir_path + "/ipknot"
  knotty_dir_path = asset_dir_path + "/knotty"
  spot_rna_dir_path = asset_dir_path + "/spot_rna"
  raf_dir_path = asset_dir_path + "/raf"
  locarna_dir_path = asset_dir_path + "/locarna"
  turbofold_dir_path = asset_dir_path + "/turbofold"
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  if not os.path.isdir(ipknot_dir_path):
    os.mkdir(ipknot_dir_path)
  if not os.path.isdir(knotty_dir_path):
    os.mkdir(knotty_dir_path)
  if not os.path.isdir(spot_rna_dir_path):
    os.mkdir(spot_rna_dir_path)
  if not os.path.isdir(raf_dir_path):
    os.mkdir(raf_dir_path)
  if not os.path.isdir(locarna_dir_path):
    os.mkdir(locarna_dir_path)
  if not os.path.isdir(turbofold_dir_path):
    os.mkdir(turbofold_dir_path)
  rna_dir_path = asset_dir_path + "/test_data"
  sub_thread_num = 4
  for rna_file in os.listdir(rna_dir_path):
    if not rna_file.endswith(".fa"):
      continue
    rna_file_path = os.path.join(rna_dir_path, rna_file)
    (rna_family_name, extension) = os.path.splitext(rna_file)
    ipknot_output_dir_path = os.path.join(ipknot_dir_path, rna_family_name)
    knotty_output_file_path = os.path.join(knotty_dir_path, rna_family_name + ".fa")
    spot_rna_output_file_path = os.path.join(spot_rna_dir_path, rna_family_name + ".fa")
    raf_output_file_path = os.path.join(raf_dir_path, rna_family_name + ".fa")
    locarna_output_file_path = os.path.join(locarna_dir_path, rna_family_name + ".sth")
    turbofold_output_dir_path = os.path.join(turbofold_dir_path, rna_family_name)
    infernal_black_list_file_path = os.path.join(infernal_black_list_dir_path, rna_family_name + "_infernal.dat")
    if os.path.isfile(infernal_black_list_file_path):
      continue
    if not os.path.isdir(ipknot_output_dir_path):
      os.mkdir(ipknot_output_dir_path)
    if not os.path.isdir(turbofold_output_dir_path):
      os.mkdir(turbofold_output_dir_path)
    knotty_params.insert(0, (rna_file_path, knotty_output_file_path))
    spot_rna_params.insert(0, (rna_file_path, spot_rna_output_file_path))
    raf_params.insert(0, (rna_file_path, raf_output_file_path))
    locarna_params.insert(0, (rna_file_path, locarna_output_file_path))
    for gamma in gammas:
      gamma_str = str(gamma) if gamma < 1 else str(int(gamma))
      output_file = "gamma=" + gamma_str + ".fa"
      ipknot_output_file_path = os.path.join(ipknot_output_dir_path, output_file)
      ipknot_params.insert(0, (rna_file_path, ipknot_output_file_path, gamma_str))
      turbofold_output_file_path = os.path.join(turbofold_output_dir_path, output_file)
      turbofold_params.insert(0, (rna_file_path, turbofold_output_file_path, gamma, temp_dir_path, rna_family_name))
  pool = multiprocessing.Pool(int(num_of_threads / sub_thread_num))
  pool = multiprocessing.Pool(num_of_threads)
  # pool.map(run_ipknot, ipknot_params)
  # pool.map(run_knotty, knotty_params)
  # pool.map(run_spot_rna, spot_rna_params)
  # pool.map(run_raf, raf_params)
  # pool.map(run_locarna, locarna_params)
  pool.map(run_turbofold, turbofold_params)
  shutil.rmtree(temp_dir_path)

def run_ipknot(ipknot_params):
  (rna_file_path, ipknot_output_file_path) = ipknot_params
  ipknot_command = "ipknot " + rna_file_path
  (output, _, _) = utils.run_command(rnafold_command)
  lines = [line.split()[0] for (i, line) in enumerate(str(output).split("\\n")) if i % 3 == 2]
  ipknot_output_file = open(ipknot_output_file_path, "w+")
  ipknot_output_buf = ""
  for (i, line) in enumerate(lines):
    ipknot_output_buf += ">%d\n%s\n\n" % (i, line)
  ipknot_output_file.write(ipknot_output_buf)
  ipknot_output_file.close()

def run_locarna(locarna_params):
  (rna_file_path, locarna_output_file_path) = ipknot_params
  locarna_command = "mlocarna " + rna_file_path + " --keep-sequence-order"
  (output, _, _) = utils.run_command(mlocarna_command)
  lines = [line.strip() for (i, line) in enumerate(str(output).split("\\n")) if i > 7]
  locarna_output_file = open(locarna_output_file_path, "w+")
  locarna_output_buf = "# STOCKHOLM 1.0\n\n"
  for line in lines:
    if line.startswith("alifold "):
      locarna_output_buf += "#=GC SS_cons %s//" % lines.split()[1]
    else:
      locarna_output_buf += line + "\n"
  locarna_output_file.write(locarna_output_buf)
  locarna_output_file.close()

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

if __name__ == "__main__":
  main()
