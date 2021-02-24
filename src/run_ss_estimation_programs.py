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
  conshomfold_params = []
  centroidhomfold_params = []
  rnafold_params = []
  contrafold_params = []
  centroidfold_params = []
  contextfold_params = []
  conshomfold_dir_path = asset_dir_path + "/conshomfold"
  centroidhomfold_dir_path = asset_dir_path + "/centroidhomfold"
  rnafold_dir_path = asset_dir_path + "/rnafold"
  contrafold_dir_path = asset_dir_path + "/contrafold"
  centroidfold_dir_path = asset_dir_path + "/centroidfold"
  contextfold_dir_path = asset_dir_path + "/contextfold"
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  if not os.path.isdir(conshomfold_dir_path):
    os.mkdir(conshomfold_dir_path)
  if not os.path.isdir(centroidhomfold_dir_path):
    os.mkdir(centroidhomfold_dir_path)
  if not os.path.isdir(rnafold_dir_path):
    os.mkdir(rnafold_dir_path)
  if not os.path.isdir(contrafold_dir_path):
    os.mkdir(contrafold_dir_path)
  if not os.path.isdir(centroidfold_dir_path):
    os.mkdir(centroidfold_dir_path)
  if not os.path.isdir(contextfold_dir_path):
    os.mkdir(contextfold_dir_path)
  rna_dir_path = asset_dir_path + "/test_data"
  sub_thread_num = 4
  for rna_file in os.listdir(rna_dir_path):
    if not rna_file.endswith(".fa"):
      continue
    rna_file_path = os.path.join(rna_dir_path, rna_file)
    (rna_family_name, extension) = os.path.splitext(rna_file)
    conshomfold_output_dir_path = os.path.join(conshomfold_dir_path, rna_family_name)
    centroidhomfold_output_dir_path = os.path.join(centroidhomfold_dir_path, rna_family_name)
    rnafold_output_file_path = os.path.join(rnafold_dir_path, rna_family_name + ".fa")
    contrafold_output_dir_path = os.path.join(contrafold_dir_path, rna_family_name)
    contextfold_output_file_path = os.path.join(contextfold_dir_path, rna_family_name + ".fa")
    centroidfold_output_dir_path = os.path.join(centroidfold_dir_path, rna_family_name)
    infernal_black_list_file_path = os.path.join(infernal_black_list_dir_path, rna_family_name + "_infernal.dat")
    if os.path.isfile(infernal_black_list_file_path):
      continue
    if not os.path.isdir(conshomfold_output_dir_path):
      os.mkdir(conshomfold_output_dir_path)
    if not os.path.isdir(centroidhomfold_output_dir_path):
      os.mkdir(centroidhomfold_output_dir_path)
    if not os.path.isdir(contrafold_output_dir_path):
      os.mkdir(contrafold_output_dir_path)
    if not os.path.isdir(centroidfold_output_dir_path):
      os.mkdir(centroidfold_output_dir_path)
    conshomfold_command = "conshomfold -t " + str(sub_thread_num) + " -i " + rna_file_path + " -o " + conshomfold_output_dir_path
    conshomfold_params.insert(0, conshomfold_command)
    rnafold_params.insert(0, (rna_file_path, rnafold_output_file_path))
    for gamma in gammas:
      gamma_str = str(gamma) if gamma < 1 else str(int(gamma))
      output_file = "gamma=" + gamma_str + ".fa"
      centroidhomfold_output_file_path = os.path.join(centroidhomfold_output_dir_path, output_file)
      centroidhomfold_params.insert(0, (rna_file_path, centroidhomfold_output_file_path, gamma_str, temp_dir_path))
      contrafold_output_file_path = os.path.join(contrafold_output_dir_path, output_file)
      contrafold_params.insert(0, (rna_file_path, contrafold_output_file_path, gamma_str, temp_dir_path))
      contextfold_params.insert(0, (rna_file_path, contextfold_output_file_path, temp_dir_path))
      centroidfold_output_file_path = os.path.join(centroidfold_output_dir_path, output_file)
      centroidfold_params.insert(0, (rna_file_path, centroidfold_output_file_path, gamma_str))
  pool = multiprocessing.Pool(int(num_of_threads / sub_thread_num))
  # pool.map(utils.run_command, conshomfold_params)
  pool = multiprocessing.Pool(num_of_threads)
  # pool.map(run_centroidhomfold, centroidhomfold_params)
  # pool.map(run_rnafold, rnafold_params)
  # pool.map(run_contrafold, contrafold_params)
  pool.map(run_contextfold, contextfold_params)
  # pool.map(run_centroidfold, centroidfold_params)
  shutil.rmtree(temp_dir_path)

def run_rnafold(rnafold_params):
  (rna_file_path, rnafold_output_file_path) = rnafold_params
  rnafold_command = "RNAfold --noPS -i " + rna_file_path
  (output, _, _) = utils.run_command(rnafold_command)
  lines = [str(line) for line in str(output).strip().split("\\n") if line.startswith(".") or line.startswith("(")]
  buf = ""
  for i, line in enumerate(lines):
    buf += ">%d\n%s\n" % (i, line.split()[0])
  rnafold_output_file = open(rnafold_output_file_path, "w")
  rnafold_output_file.write(buf)
  rnafold_output_file.close()

def run_centroidfold(centroidfold_params):
  (rna_file_path, centroidfold_output_file_path, gamma_str) = centroidfold_params
  centroidfold_command = "centroid_fold " + rna_file_path + " -g " + gamma_str
  (output, _, _) = utils.run_command(centroidfold_command)
  lines = [line.split()[0] for (i, line) in enumerate(str(output).split("\\n")) if i % 3 == 2]
  centroidfold_output_file = open(centroidfold_output_file_path, "w+")
  centroidfold_output_buf = ""
  for (i, line) in enumerate(lines):
    centroidfold_output_buf += ">%d\n%s\n\n" % (i, line)
  centroidfold_output_file.write(centroidfold_output_buf)
  centroidfold_output_file.close()

def run_contrafold(contrafold_params):
  (rna_file_path, contrafold_output_file_path, gamma_str, temp_dir_path) = contrafold_params
  contrafold_output_file = open(contrafold_output_file_path, "w+")
  contrafold_output_buf = ""
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  seq_file_path = os.path.join(temp_dir_path, "seq_4_%s_and_gamma=%s.fa" % (rna_family_name, gamma_str))
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  for (i, rec) in enumerate(recs):
    SeqIO.write([rec], open(seq_file_path, "w"), "fasta")
    contrafold_command = "contrafold predict " + seq_file_path + " --gamma " + gamma_str
    (output, _, _) = utils.run_command(contrafold_command)
    contrafold_output_buf += ">%d\n%s\n\n" % (i, str(output).strip().split("\\n")[3])
  contrafold_output_file.write(contrafold_output_buf)
  contrafold_output_file.close()

def run_centroidhomfold(centroidhomfold_params):
  (rna_file_path, centroidhomfold_output_file_path, gamma_str, temp_dir_path) = centroidhomfold_params
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  rec_seq_len = len(recs)
  centroidhomfold_output_file = open(centroidhomfold_output_file_path, "w+")
  centroidhomfold_output_buf = ""
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  seq_file_path = os.path.join(temp_dir_path, "seqs_4_%s_and_gamma=%s.fa" % (rna_family_name, gamma_str))
  hom_seq_file_path = os.path.join(temp_dir_path, "hom_seqs_4_%s_and_gamma=%s.fa" % (rna_family_name, gamma_str))
  for (i, rec) in enumerate(recs):
    SeqIO.write([rec], open(seq_file_path, "w"), "fasta")
    hom_recs = [rec for (j, rec) in enumerate(recs) if j != i]
    SeqIO.write(recs, open(hom_seq_file_path, "w"), "fasta")
    centroidhomfold_command = "centroid_homfold " + seq_file_path + " -H " + hom_seq_file_path + " -g " + gamma_str
    (output, _, _) = utils.run_command(centroidhomfold_command)
    centroidhomfold_output_buf += ">%d\n%s\n\n" % (i, str(output).split("\\n")[2].split()[0])
  centroidhomfold_output_file.write(centroidhomfold_output_buf)
  centroidhomfold_output_file.close()

def run_contextfold(contextfold_params):
  (rna_file_path, contextfold_output_file_path, temp_dir_path) = contextfold_params
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  seq_file_path = os.path.join(temp_dir_path, "seqs_4_%s.txt" % rna_family_name)
  tmp_output_buf = ""
  for rec in recs:
    tmp_output_buf += str(rec.seq) + "\n"
  seq_file = open(seq_file_path, "w+")
  seq_file.write(tmp_output_buf)
  seq_file.close()
  contextfold_command = "java -cp bin contextFold.app.Predict in:" + seq_file_path
  (_, _, _) = utils.run_command(contextfold_command)
  with open(seq_file_path + ".pred") as f:
    sss = [line.strip() for (i, line) in enumerate(f.readlines()) if (i + 1) % 3 == 2]
  contextfold_output_buf = ""
  for (i, ss) in enumerate(sss):
    contextfold_output_buf += ">%d\n%s\n\n" % (i, ss)
  contextfold_output_file = open(contextfold_output_file_path, "w+")
  contextfold_output_file.write(contextfold_output_buf)
  contextfold_output_file.close()

if __name__ == "__main__":
  main()
