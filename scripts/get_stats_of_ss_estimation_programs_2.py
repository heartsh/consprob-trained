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
  consfold_ppvs = []
  consfold_senss = []
  consfold_fprs = []
  consfold_f1_scores = []
  consfold_mccs = []
  linearfold_ppv = 0
  linearfold_sens = 0
  linearfold_fpr = 0
  linearfold_f1_score = 0
  linearfold_mcc = 0
  mxfold_ppv = 0
  mxfold_sens = 0
  mxfold_fpr = 0
  mxfold_f1_score = 0
  mxfold_mcc = 0
  probknot_ppv = 0
  probknot_sens = 0
  probknot_fpr = 0
  probknot_f1_score = 0
  probknot_mcc = 0
  ipknot_ppv = 0
  ipknot_sens = 0
  ipknot_fpr = 0
  ipknot_f1_score = 0
  ipknot_mcc = 0
  spot_rna_ppv = 0
  spot_rna_sens = 0
  spot_rna_fpr = 0
  spot_rna_f1_score = 0
  spot_rna_mcc = 0
  gammas = [2. ** i for i in range(-7, 11)]
  consfold_ss_dir_path = asset_dir_path + "/consfold"
  linearfold_ss_dir_path = asset_dir_path + "/linearfold"
  mxfold_ss_dir_path = asset_dir_path + "/mxfold"
  probknot_ss_dir_path = asset_dir_path + "/probknot"
  ipknot_ss_dir_path = asset_dir_path + "/ipknot"
  spot_rna_ss_dir_path = asset_dir_path + "/spot_rna"
  rna_fam_dir_path = asset_dir_path + "/test_ref_sss"
  pool = multiprocessing.Pool(num_of_threads)
  for gamma in gammas:
    consfold_count_params = []
    linearfold_count_params = []
    mxfold_count_params = []
    probknot_count_params = []
    ipknot_count_params = []
    spot_rna_count_params = []
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
      consfold_estimated_ss_dir_path = os.path.join(consfold_ss_dir_path, rna_fam_name)
      consfold_estimated_ss_file_path = os.path.join(consfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      consfold_count_params.insert(0, (consfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
      if gamma == 1.:
        linearfold_estimated_ss_file_path = os.path.join(linearfold_ss_dir_path, rna_fam_name + ".fa")
        linearfold_count_params.insert(0, (linearfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
        mxfold_estimated_ss_file_path = os.path.join(mxfold_ss_dir_path, rna_fam_name + ".fa")
        mxfold_count_params.insert(0, (mxfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
        probknot_estimated_ss_file_path = os.path.join(probknot_ss_dir_path, rna_fam_name + ".bpseq")
        probknot_count_params.insert(0, (probknot_estimated_ss_file_path, ref_sss, rna_seq_lens))
        ipknot_estimated_ss_dir_path = os.path.join(ipknot_ss_dir_path, )
        ipknot_estimated_ss_file_path = os.path.join(ipknot_ss_dir_path, rna_fam_name + ".fa")
        ipknot_count_params.insert(0, (ipknot_estimated_ss_file_path, ref_sss, rna_seq_lens))
        spot_rna_estimated_ss_file_path = os.path.join(spot_rna_ss_dir_path, rna_fam_name + ".bpseq")
        spot_rna_count_params.insert(0, (spot_rna_estimated_ss_file_path, ref_sss, rna_seq_lens))
    results = pool.map(get_pos_neg_counts, consfold_count_params)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    consfold_ppvs.insert(0, ppv)
    consfold_senss.insert(0, sens)
    consfold_fprs.insert(0, fpr)
    consfold_f1_scores.append(get_f1_score(ppv, sens))
    consfold_mccs.append(get_mcc(tp, tn, fp, fn))
    if gamma == 1.:
      results = pool.map(get_pos_neg_counts, linearfold_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      linearfold_ppv = ppv
      linearfold_sens = sens
      linearfold_fpr = fpr
      linearfold_f1_score = get_f1_score(ppv, sens)
      linearfold_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_pos_neg_counts, mxfold_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      mxfold_ppv = ppv
      mxfold_sens = sens
      mxfold_fpr = fpr
      mxfold_f1_score = get_f1_score(ppv, sens)
      mxfold_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_pos_neg_counts, probknot_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      probknot_ppv = ppv
      probknot_sens = sens
      probknot_fpr = fpr
      probknot_f1_score = get_f1_score(ppv, sens)
      probknot_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_pos_neg_counts, ipknot_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      ipknot_ppv = ppv
      ipknot_sens = sens
      ipknot_fpr = fpr
      ipknot_f1_score = get_f1_score(ppv, sens)
      ipknot_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_pos_neg_counts, spot_rna_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      spot_rna_ppv = ppv
      spot_rna_sens = sens
      spot_rna_fpr = fpr
      spot_rna_f1_score = get_f1_score(ppv, sens)
      spot_rna_mcc = get_mcc(tp, tn, fp, fn)
  line_1, = pyplot.plot(consfold_ppvs, consfold_senss, label = "ConsFold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(linearfold_ppv, linearfold_sens, label = "LinearFold", marker = "s")
  line_3, = pyplot.plot(mxfold_ppv, mxfold_sens, label = "MXfold2", marker = "*")
  line_4, = pyplot.plot(probknot_ppv, probknot_sens, label = "ProbKnot", marker = "p")
  line_5, = pyplot.plot(ipknot_ppv, ipknot_sens, label = "IPknot", marker = "D")
  line_6, = pyplot.plot(spot_rna_ppv, spot_rna_sens, label = "SPOT-RNA", marker = "v")
  pyplot.xlabel("Precision")
  pyplot.ylabel("Recall")
  pyplot.legend(handles = [line_1, line_2, line_3, line_4, line_5, line_6], loc = "lower left")
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/pr_curves_on_ss_estimation_2.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(consfold_fprs, consfold_senss, label = "ConsFold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(linearfold_fpr, linearfold_sens, label = "LinearFold", marker = "s")
  line_3, = pyplot.plot(mxfold_fpr, mxfold_sens, label = "MXfold2", marker = "*")
  line_4, = pyplot.plot(probknot_fpr, probknot_sens, label = "ProbKnot", marker = "p")
  line_5, = pyplot.plot(ipknot_fpr, ipknot_sens, label = "IPknot", marker = "D")
  line_6, = pyplot.plot(spot_rna_fpr, spot_rna_sens, label = "SPOT-RNA", marker = "v")
  pyplot.xlabel("Fall-out")
  pyplot.ylabel("Recall")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/roc_curves_on_ss_estimation_2.eps", bbox_inches = "tight")
  pyplot.clf()
  gammas = [i for i in range(-7, 11)]
  line_1, = pyplot.plot(gammas, consfold_f1_scores, label = "ConsFold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(-2, linearfold_f1_score, label = "LinearFold", marker = "s")
  line_3, = pyplot.plot(-2, mxfold_f1_score, label = "MXfold2", marker = "*")
  line_4, = pyplot.plot(-2, probknot_f1_score, label = "ProbKnot", marker = "p")
  line_5, = pyplot.plot(-2, ipknot_f1_score, label = "IPknot", marker = "D")
  line_6, = pyplot.plot(-2, spot_rna_f1_score, label = "SPOT-RNA", marker = "v")
  line_7, = pyplot.plot(min_gamma + numpy.argmax(consfold_f1_scores), max(consfold_f1_scores), label = "ConsFold", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("F1 score")
  pyplot.legend(handles = [line_7], loc = "lower right")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_f1_scores_on_ss_estimation_2.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(gammas, consfold_mccs, label = "ConsFold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(-2, linearfold_mcc, label = "LinearFold", marker = "s")
  line_3, = pyplot.plot(-2, mxfold_mcc, label = "MXfold2", marker = "*")
  line_4, = pyplot.plot(-2, probknot_mcc, label = "ProbKnot", marker = "p")
  line_5, = pyplot.plot(-2, ipknot_mcc, label = "IPknot", marker = "D")
  line_6, = pyplot.plot(-2, spot_rna_mcc, label = "SPOT-RNA", marker = "v")
  line_7, = pyplot.plot(min_gamma + numpy.argmax(consfold_mccs), max(consfold_mccs), label = "ConsFold", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("MCC")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_mccs_on_ss_estimation_2.eps", bbox_inches = "tight")

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
