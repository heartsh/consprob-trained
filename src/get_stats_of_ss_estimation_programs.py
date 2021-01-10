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
  num_of_threads = multiprocessing.cpu_count()
  centroid_estimator_ppvs = []
  centroid_estimator_senss = []
  centroid_estimator_fprs = []
  centroid_estimator_f1_scores = []
  centroid_estimator_mccs = []
  conshomfold_ppvs = []
  conshomfold_senss = []
  conshomfold_fprs = []
  conshomfold_f1_scores = []
  conshomfold_mccs = []
  gammas = [2. ** i for i in range(-7, 11)]
  centroid_estimator_ss_dir_path = asset_dir_path + "/centroid_estimator"
  conshomfold_ss_dir_path = asset_dir_path + "/conshomfold"
  rna_fam_dir_path = asset_dir_path + "/test_ref_sss"
  pool = multiprocessing.Pool(num_of_threads)
  for gamma in gammas:
    centroid_estimator_count_params = []
    conshomfold_count_params = []
    gamma_str = str(gamma) if gamma < 1 else str(int(gamma))
    for rna_fam_file in os.listdir(rna_fam_dir_path):
      if not rna_fam_file.endswith(".fa"):
        continue
      rna_seq_file_path = os.path.join(rna_fam_dir_path, rna_fam_file)
      rna_seq_lens = [len(rna_seq.seq) for rna_seq in SeqIO.parse(rna_seq_file_path, "fasta")]
      (rna_fam_name, extension) = os.path.splitext(rna_fam_file)
      ref_ss_file_path = os.path.join(rna_fam_dir_path, rna_fam_file)
      ref_sss_and_flat_sss = utils.get_sss_and_flat_sss(utils.get_ss_strings(ref_ss_file_path))
      centroid_estimator_estimated_ss_dir_path = os.path.join(centroid_estimator_ss_dir_path, rna_fam_name)
      centroid_estimator_estimated_ss_file_path = os.path.join(centroid_estimator_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      centroid_estimator_count_params.insert(0, (centroid_estimator_estimated_ss_file_path, ref_sss_and_flat_sss, rna_seq_lens))
      conshomfold_estimated_ss_dir_path = os.path.join(conshomfold_ss_dir_path, rna_fam_name)
      conshomfold_estimated_ss_file_path = os.path.join(conshomfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      conshomfold_count_params.insert(0, (conshomfold_estimated_ss_file_path, ref_sss_and_flat_sss, rna_seq_lens))
    results = pool.map(get_pos_neg_counts, centroid_estimator_count_params)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    centroid_estimator_ppvs.insert(0, ppv)
    centroid_estimator_senss.insert(0, sens)
    centroid_estimator_fprs.insert(0, fpr)
    centroid_estimator_f1_scores.append(get_f1_score(ppv, sens))
    centroid_estimator_mccs.append(get_mcc(tp, tn, fp, fn))
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
  line_1, = pyplot.plot(centroid_estimator_ppvs, centroid_estimator_senss, label = "Centroid estimator", marker = "o", linestyle = "-")
  line_2, = pyplot.plot(conshomfold_ppvs, conshomfold_senss, label = "ConsHomfold", marker = "s", linestyle = ":")
  pyplot.xlabel("Precision")
  pyplot.ylabel("Recall")
  pyplot.legend(handles = [line_1, line_2], loc = "lower left")
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/pr_curves_on_ss_estimation.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(centroid_estimator_fprs, centroid_estimator_senss, label = "Centroid estimator", marker = "o", linestyle = "-")
  line_2, = pyplot.plot(conshomfold_fprs, conshomfold_senss, label = "ConsHomfold", marker = "s", linestyle = ":")
  pyplot.xlabel("Fall-out")
  pyplot.ylabel("Recall")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/roc_curves_on_ss_estimation.eps", bbox_inches = "tight")
  pyplot.clf()
  gammas = [i for i in range(-7, 11)]
  line_1, = pyplot.plot(gammas, centroid_estimator_f1_scores, label = "Centroid estimator", marker = "o", linestyle = "-")
  line_2, = pyplot.plot(gammas, conshomfold_f1_scores, label = "ConsHomfold", marker = "s", linestyle = ":")
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("F1 score")
  pyplot.legend(handles = [line_1, line_2], loc = "lower right")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_f1_scores_on_ss_estimation.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(gammas, centroid_estimator_mccs, label = "Centroid estimator", marker = "o", linestyle = "-")
  line_2, = pyplot.plot(gammas, conshomfold_mccs, label = "ConsHomfold", marker = "s", linestyle = ":")
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("MCC")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_mccs_on_ss_estimation.eps", bbox_inches = "tight")

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
