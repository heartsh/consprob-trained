#! /usr/bin/env python

import utils
from Bio import SeqIO
import seaborn
from matplotlib import pyplot
import os
from sklearn.metrics import roc_curve
import math
from math import sqrt
import multiprocessing

seaborn.set()
color_palette = seaborn.color_palette()
color_palette_2 = seaborn.color_palette("Set2")
white = "#F2F2F2"
pyplot.rcParams["figure.dpi"] = 600

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  costs, accs = read_train_logs(asset_dir_path + "/train_logs.dat")
  epochs = [i for i in range(1, len(costs) + 1)];
  line_1, = pyplot.plot(epochs, costs, label = 'Transferred from CONTRAfold & CONTRAlign', marker = "", linestyle = "solid")
  costs_random_init, accs_random_init = read_train_logs(asset_dir_path + "/train_logs_random_init.dat")
  epochs_random_init = [i for i in range(1, len(costs_random_init) + 1)];
  line_2, = pyplot.plot(epochs_random_init, costs_random_init, label = 'Initialized to random values', marker = "", linestyle = "solid")
  pyplot.xlabel("Epoch")
  pyplot.ylabel("Cost")
  pyplot.legend(handles = [line_1, line_2], loc = "upper right")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/epochs_vs_costs.svg", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(epochs, accs, label = 'Transferred from CONTRAfold & CONTRAlign', marker = "", linestyle = "solid")
  line_2, = pyplot.plot(epochs_random_init, accs_random_init, label = 'Initialized to random values', marker = "", linestyle = "solid")
  pyplot.xlabel("Epoch")
  pyplot.ylabel("Average expected SPS")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/epochs_vs_accs.svg", bbox_inches = "tight")

def read_train_logs(train_log_file_path):
  train_log_file = open(train_log_file_path, "r")
  lines = [line.split(",") for line in train_log_file.readlines() if len(line) > 0]
  costs = list(map(lambda x: float(x[0]), lines))
  accs = list(map(lambda x: float(x[1]), lines))
  return costs, accs

def get_pos_neg_counts(params):
  (estimated_ss_file_path, ref_sss_and_flat_sss, rna_seq_lens) = params
  tp = tn = fp = fn = 0
  estimated_sss_and_flat_sss = utils.get_sss_and_flat_sss(utils.get_ss_strings(estimated_ss_file_path))
  for (estimated_ss_and_flat_ss, ref_ss_and_flat_ss, rna_seq_len) in zip(estimated_sss_and_flat_sss, ref_sss_and_flat_sss, rna_seq_lens):
    estimated_ss, estimated_flat_ss = estimated_ss_and_flat_ss
    ref_ss, ref_flat_ss = ref_ss_and_flat_ss
    for i in range(0, rna_seq_len):
      for j in range(i + 1, rna_seq_len):
        estimated_bin = (i, j) in estimated_ss
        ref_bin = (i, j) in ref_ss
        if estimated_bin == ref_bin:
          if estimated_bin == True:
            tp += 1
          else:
            tn += 1
        else:
          if estimated_bin == True:
            fp += 1
          else:
            fn += 1
  return tp, tn, fp, fn

def final_sum(results):
  final_tp = final_tn = final_fp = final_fn = 0.
  for tp, tn, fp, fn in results:
    final_tp += tp
    final_tn += tn
    final_fp += fp
    final_fn += fn
  return (final_tp, final_tn, final_fp, final_fn)

def get_f1_score(ppv, sens):
  return 2 * ppv * sens / (ppv + sens)

def get_mcc(tp, tn, fp, fn):
  return (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

def get_ppv(tp, fp):
  return tp / (tp + fp)

def get_sens(tp, fn):
  return tp / (tp + fn)

def get_fpr(tn, fp):
  return fp / (tn + fp)

if __name__ == "__main__":
  main()
