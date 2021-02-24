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
  mxfold_params = []
  linearfold_params = []
  probknot_params = []
  ipknot_params = []
  spot_rna_params = []
  linearfold_dir_path = asset_dir_path + "/linearfold"
  mxfold_dir_path = asset_dir_path + "/mxfold"
  probknot_dir_path = asset_dir_path + "/probknot"
  ipknot_dir_path = asset_dir_path + "/ipknot"
  spot_rna_dir_path = asset_dir_path + "/spot_rna"
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  if not os.path.isdir(mxfold_dir_path):
    os.mkdir(mxfold_dir_path)
  if not os.path.isdir(linearfold_dir_path):
    os.mkdir(linearfold_dir_path)
  if not os.path.isdir(probknot_dir_path):
    os.mkdir(probknot_dir_path)
  if not os.path.isdir(ipknot_dir_path):
    os.mkdir(ipknot_dir_path)
  if not os.path.isdir(spot_rna_dir_path):
    os.mkdir(spot_rna_dir_path)
  rna_dir_path = asset_dir_path + "/test_data"
  sub_thread_num = 4
  for rna_file in os.listdir(rna_dir_path):
    if not rna_file.endswith(".fa"):
      continue
    rna_file_path = os.path.join(rna_dir_path, rna_file)
    (rna_family_name, extension) = os.path.splitext(rna_file)
    linearfold_output_file_path = os.path.join(linearfold_dir_path, rna_family_name + ".fa")
    mxfold_output_file_path = os.path.join(mxfold_dir_path, rna_family_name + ".fa")
    ipknot_output_file_path = os.path.join(ipknot_dir_path, rna_family_name + ".fa")
    probknot_output_file_path = os.path.join(probknot_dir_path, rna_family_name + ".bpseq")
    spot_rna_output_file_path = os.path.join(spot_rna_dir_path, rna_family_name + ".bpseq")
    infernal_black_list_file_path = os.path.join(infernal_black_list_dir_path, rna_family_name + "_infernal.dat")
    if os.path.isfile(infernal_black_list_file_path):
      continue
    linearfold_params.insert(0, (rna_file_path, linearfold_output_file_path))
    mxfold_params.insert(0, (rna_file_path, mxfold_output_file_path))
    probknot_params.insert(0, (rna_file_path, probknot_output_file_path, temp_dir_path))
    spot_rna_params.insert(0, (rna_file_path, spot_rna_output_file_path, temp_dir_path))
    ipknot_params.insert(0, (rna_file_path, ipknot_output_file_path))
  pool = multiprocessing.Pool(num_of_threads)
  # pool.map(run_linearfold, linearfold_params)
  # pool.map(run_mxfold, mxfold_params)
  # pool.map(run_probknot, probknot_params)
  # pool.map(run_ipknot, ipknot_params)
  # pool.map(run_spot_rna, spot_rna_params)
  shutil.rmtree(temp_dir_path)

def run_linearfold(linearfold_params):
  (rna_file_path, linearfold_output_file_path) = linearfold_params
  linearfold_command = "cat " + rna_file_path + " | linearfold"
  (output, _, _) = utils.run_command(linearfold_command)
  lines = [str(line).split()[0] for (i, line) in enumerate(str(output).strip().split("\\n")) if (i + 1) % 3 == 0]
  buf = ""
  for i, line in enumerate(lines):
    buf += ">%d\n%s\n" % (i, line)
  linearfold_output_file = open(linearfold_output_file_path, "w")
  linearfold_output_file.write(buf)
  linearfold_output_file.close()

def run_mxfold(mxfold_params):
  (rna_file_path, mxfold_output_file_path) = mxfold_params
  mxfold_command = "mxfold2 predict " + rna_file_path
  (output, _, _) = utils.run_command(mxfold_command)
  lines = [line.split()[0] for (i, line) in enumerate(str(output).split("\\n")) if i % 3 == 2]
  mxfold_output_file = open(mxfold_output_file_path, "w+")
  mxfold_output_buf = ""
  for (i, line) in enumerate(lines):
    mxfold_output_buf += ">%d\n%s\n\n" % (i, line)
  mxfold_output_file.write(mxfold_output_buf)
  mxfold_output_file.close()

def run_probknot(probknot_params):
  (rna_file_path, probknot_output_file_path, temp_dir_path) = probknot_params
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  output_buf = ""
  seq_file_path = os.path.join(temp_dir_path, "seqs_4_%s.fa" % rna_family_name)
  for rec in recs:
    with open(seq_file_path, "w") as f:
      SeqIO.write(rec, f, "fasta")
    probknot_command = "ProbKnot " + seq_file_path + " " + probknot_output_file_path + " --sequence"
    (_, _, _) = utils.run_command(probknot_command)
    with open(probknot_output_file_path) as f:
      for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
          continue
        split = line.split()
        split_len = len(split)
        if split_len == 2:
          output_buf += "# " + line + "\n"
        else:
          output_buf += "%s %s %s\n" % (split[0], split[1], split[4])
  with open(probknot_output_file_path, "w") as f:
    f.write(output_buf)

def run_ipknot(ipknot_params):
  (rna_file_path, ipknot_output_file_path) = ipknot_params
  ipknot_command = "ipknot " + rna_file_path
  (output, _, _) = utils.run_command(ipknot_command)
  lines = [line.split()[0] for (i, line) in enumerate(str(output).split("\\n")) if i % 3 == 2]
  ipknot_output_file = open(ipknot_output_file_path, "w+")
  ipknot_output_buf = ""
  for (i, line) in enumerate(lines):
    ipknot_output_buf += ">%d\n%s\n\n" % (i, line)
  ipknot_output_file.write(ipknot_output_buf)
  ipknot_output_file.close()

def run_spot_rna(spot_rna_params):
  (rna_file_path, spot_rna_output_file_path, temp_dir_path) = spot_rna_params
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  for (i, rec) in enumerate(recs):
    new_rec = rec
    new_rec.id = str(i)
    new_rec.name = str(i)
    new_rec.description = str(i)
    recs[i] = new_rec
  spot_rna_temp_dir_path = temp_dir_path + "/" + rna_family_name
  if not os.path.isdir(spot_rna_temp_dir_path):
    os.mkdir(spot_rna_temp_dir_path)
  seq_file_path = os.path.join(temp_dir_path, "seqs_4_%s.fa" % rna_family_name)
  SeqIO.write(recs, open(seq_file_path, "w"), "fasta")
  spot_rna_command = "python3 $(which SPOT-RNA.py) --inputs " + seq_file_path + " --outputs " + spot_rna_temp_dir_path + " --cpu 1"
  (_, _, _) = utils.run_command(spot_rna_command)
  output_buf = ""
  for i, rec in enumerate(recs):
    with open(spot_rna_temp_dir_path + "/" + rec.id + ".bpseq") as f:
      for line in f.readlines():
        output_buf += line
  with open(spot_rna_output_file_path, "w") as f:
    f.write(output_buf)

if __name__ == "__main__":
  main()
