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
import numpy

seaborn.set()
pyplot.rcParams['legend.handlelength'] = 0
pyplot.rcParams['legend.fontsize'] = "x-large"
color_palette = seaborn.color_palette()
min_gamma = -7
max_gamma = 10
white = "#F2F2F2"

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  conshomfold_ppvs = []
  conshomfold_senss = []
  conshomfold_fprs = []
  conshomfold_f1_scores = []
  conshomfold_mccs = []
  raf_sens = 0
  raf_ppv = 0
  raf_fpr = 0
  raf_f1_score = 0
  raf_mcc = 0
  locarna_ppv = 0
  locarna_sens = 0
  locarna_fpr = 0
  locarna_f1_score = 0
  locarna_mcc = 0
  dafs_ppv = 0
  dafs_sens = 0
  dafs_fpr = 0
  dafs_f1_score = 0
  dafs_mcc = 0
  sparse_ppv = 0
  sparse_sens = 0
  sparse_fpr = 0
  sparse_f1_score = 0
  sparse_mcc = 0
  turbofold_ppvs = []
  turbofold_senss = []
  turbofold_fprs = []
  turbofold_f1_scores = []
  turbofold_mccs = []
  gammas = [2. ** i for i in range(-7, 11)]
  conshomfold_ss_dir_path = asset_dir_path + "/conshomfold"
  raf_ss_dir_path = asset_dir_path + "/raf"
  locarna_ss_dir_path = asset_dir_path + "/locarna"
  dafs_ss_dir_path = asset_dir_path + "/dafs"
  sparse_ss_dir_path = asset_dir_path + "/sparse"
  turbofold_ss_dir_path = asset_dir_path + "/turbofold"
  rna_fam_dir_path = asset_dir_path + "/test_ref_sss"
  pool = multiprocessing.Pool(num_of_threads)
  for gamma in gammas:
    conshomfold_count_params = []
    raf_count_params = []
    locarna_count_params = []
    dafs_count_params = []
    sparse_count_params = []
    turbofold_count_params = []
    gamma_str = str(gamma) if gamma < 1 else str(int(gamma))
    for rna_fam_file in os.listdir(rna_fam_dir_path):
      if not rna_fam_file.endswith(".fa"):
        continue
      (rna_fam_name, extension) = os.path.splitext(rna_fam_file)
      infernal_black_list_file_path = os.path.join(infernal_black_list_dir_path, rna_fam_name + "_infernal.dat")
      if os.path.isfile(infernal_black_list_file_path):
        continue
      rna_seq_file_path = os.path.join(rna_fam_dir_path, rna_fam_file)
      rna_seq_lens = [len(rna_seq.seq) for rna_seq in SeqIO.parse(rna_seq_file_path, "fasta")]
      ref_ss_file_path = os.path.join(rna_fam_dir_path, rna_fam_file)
      ref_sss = utils.get_sss(ref_ss_file_path)
      conshomfold_estimated_ss_dir_path = os.path.join(conshomfold_ss_dir_path, rna_fam_name)
      conshomfold_estimated_ss_file_path = os.path.join(conshomfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".bpseq")
      conshomfold_count_params.insert(0, (conshomfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
      if gamma == 1.:
        raf_estimated_ss_file_path = os.path.join(raf_ss_dir_path, rna_fam_name + ".fa")
        raf_count_params.insert(0, (raf_estimated_ss_file_path, ref_sss, rna_seq_lens))
        locarna_estimated_ss_file_path = os.path.join(locarna_ss_dir_path, rna_fam_name + ".fa")
        locarna_count_params.insert(0, (locarna_estimated_ss_file_path, ref_sss, rna_seq_lens))
        dafs_estimated_ss_file_path = os.path.join(dafs_ss_dir_path, rna_fam_name + ".fa")
        dafs_count_params.insert(0, (dafs_estimated_ss_file_path, ref_sss, rna_seq_lens))
        sparse_estimated_ss_file_path = os.path.join(sparse_ss_dir_path, rna_fam_name + ".fa")
        sparse_count_params.insert(0, (sparse_estimated_ss_file_path, ref_sss, rna_seq_lens))
      turbofold_estimated_ss_dir_path = os.path.join(turbofold_ss_dir_path, rna_fam_name)
      conshomfold_estimated_ss_file_path = os.path.join(conshomfold_estimated_ss_dir_path, )
      turbofold_estimated_ss_file_path = os.path.join(turbofold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      turbofold_count_params.insert(0, (turbofold_estimated_ss_file_path, ref_sss, rna_seq_lens))
    results = pool.map(get_pos_neg_counts, conshomfold_count_params)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    conshomfold_ppvs.insert(0, ppv)
    conshomfold_senss.insert(0, sens)
    conshomfold_fprs.insert(0, fpr)
    conshomfold_f1_scores.append(get_f1_score(ppv, sens))
    conshomfold_mccs.append(get_mcc(tp, tn, fp, fn))
    if gamma == 1.:
      results = pool.map(get_pos_neg_counts, raf_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      raf_ppv = ppv
      raf_sens = sens
      raf_fpr = fpr
      raf_f1_score = get_f1_score(ppv, sens)
      raf_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_pos_neg_counts, locarna_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      locarna_ppv = ppv
      locarna_sens = sens
      locarna_fpr = fpr
      locarna_f1_score = get_f1_score(ppv, sens)
      locarna_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_pos_neg_counts, dafs_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      dafs_ppv = ppv
      dafs_sens = sens
      dafs_fpr = fpr
      dafs_f1_score = get_f1_score(ppv, sens)
      dafs_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_pos_neg_counts, sparse_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      sparse_ppv = ppv
      sparse_sens = sens
      sparse_fpr = fpr
      sparse_f1_score = get_f1_score(ppv, sens)
      sparse_mcc = get_mcc(tp, tn, fp, fn)
    results = pool.map(get_pos_neg_counts, turbofold_count_params)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    turbofold_ppvs.insert(0, ppv)
    turbofold_senss.insert(0, sens)
    turbofold_fprs.insert(0, fpr)
    turbofold_f1_scores.append(get_f1_score(ppv, sens))
    turbofold_mccs.append(get_mcc(tp, tn, fp, fn))
  line_1, = pyplot.plot(conshomfold_ppvs, conshomfold_senss, label = "ConsHomfold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(raf_ppv, raf_sens, label = "RAF", marker = "s")
  line_3, = pyplot.plot(locarna_ppv, locarna_sens, label = "LocARNA", marker = "*")
  line_4, = pyplot.plot(dafs_ppv, dafs_sens, label = "DAFS", marker = "p")
  line_5, = pyplot.plot(sparse_ppv, sparse_sens, label = "SPARSE", marker = "D")
  line_6, = pyplot.plot(turbofold_ppvs, turbofold_senss, label = "TurboFold", marker = "v", linestyle = "dashed")
  pyplot.xlabel("Precision")
  pyplot.ylabel("Recall")
  pyplot.legend(handles = [line_1, line_2, line_3, line_4, line_5, line_6], loc = "lower left")
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/pr_curves_on_ss_estimation_3.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(conshomfold_fprs, conshomfold_senss, label = "ConsHomfold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(raf_fpr, raf_sens, label = "RAF", marker = "s")
  line_3, = pyplot.plot(locarna_fpr, locarna_sens, label = "LocARNA", marker = "*")
  line_4, = pyplot.plot(dafs_fpr, dafs_sens, label = "DAFS", marker = "p")
  line_5, = pyplot.plot(sparse_fpr, sparse_sens, label = "SPARSE", marker = "D")
  line_6, = pyplot.plot(turbofold_fprs, turbofold_senss, label = "TurboFold", marker = "v", linestyle = "dashed")
  pyplot.xlabel("Fall-out")
  pyplot.ylabel("Recall")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/roc_curves_on_ss_estimation_3.eps", bbox_inches = "tight")
  pyplot.clf()
  gammas = [i for i in range(-7, 11)]
  line_1, = pyplot.plot(gammas, conshomfold_f1_scores, label = "ConsHomfold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(-2, raf_f1_score, label = "RAF", marker = "s")
  line_3, = pyplot.plot(-2, locarna_f1_score, label = "LocARNA", marker = "*")
  line_4, = pyplot.plot(-2, dafs_f1_score, label = "DAFS", marker = "p")
  line_5, = pyplot.plot(-2, sparse_f1_score, label = "SPARSE", marker = "D")
  line_6, = pyplot.plot(gammas, turbofold_f1_scores, label = "TurboFold", marker = "v", linestyle = "dashed")
  line_7, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_f1_scores), max(conshomfold_f1_scores), label = "ConsHomfold", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
  line_8, = pyplot.plot(min_gamma + numpy.argmax(turbofold_f1_scores), max(turbofold_f1_scores), label = "TurboFold", marker = "v", markerfacecolor = white, markeredgecolor = color_palette[5])
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("F1 score")
  pyplot.legend(handles = [line_7, line_8], loc = "lower right")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_f1_scores_on_ss_estimation_3.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(gammas, conshomfold_mccs, label = "ConsHomfold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(-2, raf_mcc, label = "RAF", marker = "s")
  line_3, = pyplot.plot(-2, locarna_mcc, label = "LocARNA", marker = "*")
  line_4, = pyplot.plot(-2, dafs_mcc, label = "DAFS", marker = "p")
  line_5, = pyplot.plot(-2, sparse_mcc, label = "SPARSE", marker = "D")
  line_6, = pyplot.plot(gammas, turbofold_mccs, label = "TurboFold", marker = "v", linestyle = "dashed")
  line_7, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_mccs), max(conshomfold_mccs), label = "ConsHomfold", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
  line_8, = pyplot.plot(min_gamma + numpy.argmax(turbofold_mccs), max(turbofold_mccs), label = "TurboFold", marker = "v", markerfacecolor = white, markeredgecolor = color_palette[5])
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("MCC")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_mccs_on_ss_estimation_3.eps", bbox_inches = "tight")

def get_pos_neg_counts(params):
  (estimated_ss_file_path, ref_sss, rna_seq_lens) = params
  tp = tn = fp = fn = 0
  estimated_sss = utils.get_sss(estimated_ss_file_path)
  for (estimated_ss, ref_ss, rna_seq_len) in zip(estimated_sss, ref_sss, rna_seq_lens):
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
