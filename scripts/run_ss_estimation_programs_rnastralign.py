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
  sub_thread_num = 1
  raf_params = []
  locarna_params = []
  dafs_params = []
  sparse_params = []
  consalign_params = []
  consalign_params_disabled_alifold = []
  consalign_params_turner = []
  consalign_params_turner_disabled_transplant = []
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
  consalign_dir_path_turner_disabled_transplant = asset_dir_path + "/consalign_rnastralign_turner_disabled_transplant"
  consalign_dir_path_trained_transfer = asset_dir_path + "/consalign_rnastralign_trained_transfer"
  consalign_dir_path_trained_random_init = asset_dir_path + "/consalign_rnastralign_trained_random_init"
  consalign_dir_path_transferred_only = asset_dir_path + "/consalign_rnastralign_transferred_only"
  consalign_elapsed_time_data_file_path = asset_dir_path + "/consalign_elapsed_time_data_rnastralign.dat"
  raf_elapsed_time_data_file_path = asset_dir_path + "/raf_elapsed_time_data_rnastralign.dat"
  locarna_elapsed_time_data_file_path = asset_dir_path + "/locarna_elapsed_time_data_rnastralign.dat"
  dafs_elapsed_time_data_file_path = asset_dir_path + "/dafs_elapsed_time_data_rnastralign.dat"
  sparse_elapsed_time_data_file_path = asset_dir_path + "/sparse_elapsed_time_data_rnastralign.dat"
  turbofold_elapsed_time_data_file_path = asset_dir_path + "/turbofold_elapsed_time_data_rnastralign.dat"
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
  if not os.path.isdir(consalign_dir_path_turner_disabled_transplant):
    os.mkdir(consalign_dir_path_turner_disabled_transplant)
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
    consalign_sub_dir_path_turner_disabled_transplant = os.path.join(consalign_dir_path_turner_disabled_transplant, rna_sub_dir)
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
    if not os.path.isdir(consalign_sub_dir_path_turner_disabled_transplant):
      os.mkdir(consalign_sub_dir_path_turner_disabled_transplant)
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
    consalign_params.insert(0, (rna_file_path, consalign_sub_dir_path, sub_thread_num, "ensemble", "trained_transfer", False, False))
    consalign_params_disabled_alifold.insert(0, (rna_file_path, consalign_sub_dir_path_disabled_alifold, sub_thread_num, "ensemble", "trained_transfer", True, False))
    consalign_params_turner.insert(0, (rna_file_path, consalign_sub_dir_path_turner, sub_thread_num, "turner", "trained_transfer", True, False))
    consalign_params_turner_disabled_transplant.insert(0, (rna_file_path, consalign_sub_dir_path_turner_disabled_transplant, sub_thread_num, "turner", "trained_transfer", True, True))
    consalign_params_trained_transfer.insert(0, (rna_file_path, consalign_sub_dir_path_trained_transfer, sub_thread_num, "trained", "trained_transfer", True, False))
    consalign_params_trained_random_init.insert(0, (rna_file_path, consalign_sub_dir_path_trained_random_init, sub_thread_num, "trained", "trained_random_init", True, False))
    consalign_params_transferred_only.insert(0, (rna_file_path, consalign_sub_dir_path_transferred_only, sub_thread_num, "trained", "transferred_only", True, False))
    turbofold_params.insert(0, (rna_file_path, turbofold_sub_dir_path))
  pool = multiprocessing.Pool(int(num_of_threads / sub_thread_num))
  time_params_consalign = list(pool.map(utils.run_consalign, consalign_params))
  utils.write_elapsed_time_data(time_params_consalign, consalign_elapsed_time_data_file_path)
  pool.map(utils.run_consalign, consalign_params_disabled_alifold)
  pool.map(utils.run_consalign, consalign_params_turner)
  pool.map(utils.run_consalign, consalign_params_turner_disabled_transplant)
  pool.map(utils.run_consalign, consalign_params_trained_transfer)
  pool.map(utils.run_consalign, consalign_params_trained_random_init)
  pool.map(utils.run_consalign, consalign_params_transferred_only)
  pool = multiprocessing.Pool(num_of_threads)
  time_params_raf = list(pool.map(utils.run_raf, raf_params))
  utils.write_elapsed_time_data(time_params_raf, raf_elapsed_time_data_file_path)
  time_params_locarna = list(pool.map(utils.run_locarna, locarna_params))
  utils.write_elapsed_time_data(time_params_locarna, locarna_elapsed_time_data_file_path)
  time_params_dafs = list(pool.map(utils.run_dafs, dafs_params))
  utils.write_elapsed_time_data(time_params_dafs, dafs_elapsed_time_data_file_path)
  time_params_sparse = list(pool.map(utils.run_locarna, sparse_params))
  utils.write_elapsed_time_data(time_params_sparse, sparse_elapsed_time_data_file_path)
  time_params_turbofold = list(pool.map(utils.run_turbofold, turbofold_params))
  utils.write_elapsed_time_data(time_params_turbofold, turbofold_elapsed_time_data_file_path)
  shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
  main()
