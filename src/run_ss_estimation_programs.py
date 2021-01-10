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

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  temp_dir_path = "/tmp/run_ss_estimation_programs_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  gammas = [2. ** i for i in range(-7, 11)]
  centroid_estimator_params = []
  conshomfold_params = []
  centroid_estimator_dir_path = asset_dir_path + "/centroid_estimator"
  conshomfold_dir_path = asset_dir_path + "/conshomfold"
  if not os.path.isdir(centroid_estimator_dir_path):
    os.mkdir(centroid_estimator_dir_path)
  if not os.path.isdir(conshomfold_dir_path):
    os.mkdir(conshomfold_dir_path)
  rna_dir_path = asset_dir_path + "/test_data"
  sub_thread_num = 4
  for rna_file in os.listdir(rna_dir_path):
    if not rna_file.endswith(".fa"):
      continue
    rna_file_path = os.path.join(rna_dir_path, rna_file)
    (rna_family_name, extension) = os.path.splitext(rna_file)
    centroid_estimator_output_dir_path = os.path.join(centroid_estimator_dir_path, rna_family_name)
    conshomfold_output_dir_path = os.path.join(conshomfold_dir_path, rna_family_name)
    if not os.path.isdir(centroid_estimator_output_dir_path):
      os.mkdir(centroid_estimator_output_dir_path)
    if not os.path.isdir(conshomfold_output_dir_path):
      os.mkdir(conshomfold_output_dir_path)
    centroid_estimator_command = "centroid_estimator -t " + str(sub_thread_num) + " -i " + rna_file_path + " -o " + centroid_estimator_output_dir_path
    centroid_estimator_params.insert(0, centroid_estimator_command)
    conshomfold_command = "conshomfold -t " + str(sub_thread_num) + " -i " + rna_file_path + " -o " + conshomfold_output_dir_path
    conshomfold_params.insert(0, conshomfold_command)
  pool = multiprocessing.Pool(int(num_of_threads / sub_thread_num))
  begin = time.time()
  pool.map(utils.run_command, centroid_estimator_params)
  centroid_estimator_elapsed_time = time.time() - begin
  begin = time.time()
  pool.map(utils.run_command, conshomfold_params)
  conshomfold_elapsed_time = time.time() - begin
  print("The elapsed time of centroid estimator = %f [s]." % centroid_estimator_elapsed_time)
  print("The elapsed time of ConsHomfold = %f [s]." % conshomfold_elapsed_time)
  shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
  main()
