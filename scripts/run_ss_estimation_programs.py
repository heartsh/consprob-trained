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
from os import path

gammas = [2. ** i for i in range(-7, 11)]

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  temp_dir_path = "/tmp/run_ss_estimation_programs_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  consfold_params = []
  consfold_params_4_running_time = []
  rnafold_params = []
  contrafold_params = []
  contrafold_params_4_running_time = []
  centroidfold_params = []
  centroidfold_params_4_running_time = []
  contextfold_params = []
  consfold_dir_path = asset_dir_path + "/consfold"
  rnafold_dir_path = asset_dir_path + "/rnafold"
  contrafold_dir_path = asset_dir_path + "/contrafold"
  centroidfold_dir_path = asset_dir_path + "/centroidfold"
  contextfold_dir_path = asset_dir_path + "/contextfold"
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  if not os.path.isdir(consfold_dir_path):
    os.mkdir(consfold_dir_path)
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
    consfold_output_dir_path = os.path.join(consfold_dir_path, rna_family_name)
    rnafold_output_file_path = os.path.join(rnafold_dir_path, rna_family_name + ".fa")
    contrafold_output_dir_path = os.path.join(contrafold_dir_path, rna_family_name)
    centroidfold_output_dir_path = os.path.join(centroidfold_dir_path, rna_family_name)
    contextfold_output_file_path = os.path.join(contextfold_dir_path, rna_family_name + ".fa")
    infernal_black_list_file_path = os.path.join(infernal_black_list_dir_path, rna_family_name + "_infernal.dat")
    if os.path.isfile(infernal_black_list_file_path):
      continue
    if not os.path.isdir(consfold_output_dir_path):
      os.mkdir(consfold_output_dir_path)
    if not os.path.isdir(contrafold_output_dir_path):
      os.mkdir(contrafold_output_dir_path)
    if not os.path.isdir(centroidfold_output_dir_path):
      os.mkdir(centroidfold_output_dir_path)
    consfold_params.insert(0, (sub_thread_num, rna_file_path, consfold_output_dir_path, False))
    consfold_params_4_running_time.insert(0, (sub_thread_num, rna_file_path, consfold_output_dir_path, True))
    rnafold_params.insert(0, (rna_file_path, rnafold_output_file_path))
    contextfold_params.insert(0, (rna_file_path, contextfold_output_file_path, temp_dir_path))
    for gamma in gammas:
      gamma_str = str(gamma) if gamma < 1 else str(int(gamma))
      output_file = "gamma=" + gamma_str + ".fa"
      contrafold_output_file_path = os.path.join(contrafold_output_dir_path, output_file)
      contrafold_params.insert(0, (rna_file_path, contrafold_output_file_path, gamma_str, temp_dir_path))
      if gamma == 1.:
        contrafold_params_4_running_time.insert(0, (rna_file_path, contrafold_output_file_path, gamma_str, temp_dir_path))
      centroidfold_output_file_path = os.path.join(centroidfold_output_dir_path, output_file)
      centroidfold_params.insert(0, (rna_file_path, centroidfold_output_file_path, gamma_str))
      if gamma == 1.:
        centroidfold_params_4_running_time.insert(0, (rna_file_path, centroidfold_output_file_path, gamma_str))
  pool = multiprocessing.Pool(num_of_threads)
  pool = multiprocessing.Pool(int(num_of_threads / sub_thread_num))
  pool.map(run_consfold, consfold_params)
  begin = time.time()
  pool.map(run_consfold, consfold_params_4_running_time)
  consfold_elapsed_time = time.time() - begin
  pool = multiprocessing.Pool(num_of_threads)
  begin = time.time()
  pool.map(run_rnafold, rnafold_params)
  rnafold_elapsed_time = time.time() - begin
  pool.map(run_contrafold, contrafold_params)
  begin = time.time()
  pool.map(run_contrafold, contrafold_params_4_running_time)
  contrafold_elapsed_time = time.time() - begin
  pool.map(run_centroidfold, centroidfold_params)
  begin = time.time()
  pool.map(run_centroidfold, centroidfold_params_4_running_time)
  centroidfold_elapsed_time = time.time() - begin
  begin = time.time()
  pool.map(run_contextfold, contextfold_params)
  contextfold_elapsed_time = time.time() - begin
  print("The elapsed time of ConsAlifold (new) = %f [s]." % consfold_elapsed_time)
  print("The elapsed time of ConsAlifold (old) = %f [s]." % consfold_old_elapsed_time)
  print("The elapsed time of RNAfold = %f [s]." % rnafold_elapsed_time)
  print("The elapsed time of CONTRAfold = %f [s]." % contrafold_elapsed_time)
  print("The elapsed time of CentroidFold = %f [s]." % centroidfold_elapsed_time)
  print("The elapsed time of Contextfold = %f [s]." % contextfold_elapsed_time)
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

def run_consfold(consfold_params):
  (sub_thread_num, rna_file_path, consfold_output_dir_path, is_benchmarked) = consfold_params
  consfold_command = "consfold %s-t " % ("-b " if is_benchmarked else "") + str(sub_thread_num) + " -i " + rna_file_path + " -o " + consfold_output_dir_path
  utils.run_command(consfold_command)

if __name__ == "__main__":
  main()
