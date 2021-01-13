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

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  log_losses_transfer_on = read_log_losses(asset_dir_path + "/log_losses.dat")
  epochs_transfer_on = [i for i in range(1, len(log_losses_transfer_on) + 1)];
  log_losses_transfer_off = read_log_losses(asset_dir_path + "/log_losses_transfer_off.dat")
  epochs_transfer_off = [i for i in range(1, len(log_losses_transfer_off) + 1)];
  line_1, = pyplot.plot(epochs_transfer_on, log_losses_transfer_on, label = "Transfer learning on", marker = "", linestyle = "-")
  line_2, = pyplot.plot(epochs_transfer_off, log_losses_transfer_off, label = "Transfer learning off", marker = "", linestyle = "--")
  pyplot.xlabel("Epoch")
  pyplot.ylabel("Cost")
  pyplot.legend(handles = [line_1, line_2], loc = "upper right")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/epochs_vs_costs.eps", bbox_inches = "tight")

def read_log_losses(log_loss_file_path):
  log_loss_file = open(log_loss_file_path, "r")
  log_losses = [float(log_loss) for log_loss in log_loss_file.readlines() if len(log_loss) > 0]
  return log_losses

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
