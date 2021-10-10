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
min_gamma = -4
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
  conshomfold_old_ppvs_turner = []
  conshomfold_old_senss_turner = []
  conshomfold_old_fprs_turner = []
  conshomfold_old_f1_scores_turner = []
  conshomfold_old_mccs_turner = []
  conshomfold_old_ppvs_contra = []
  conshomfold_old_senss_contra = []
  conshomfold_old_fprs_contra = []
  conshomfold_old_f1_scores_contra = []
  conshomfold_old_mccs_contra = []
  centroidhomfold_ppvs = []
  centroidhomfold_senss = []
  centroidhomfold_fprs = []
  centroidhomfold_f1_scores = []
  centroidhomfold_mccs = []
  if False:
    rnaalifold_ppv = 0
    rnaalifold_sens = 0
    rnaalifold_fpr = 0
    rnaalifold_f1_score = 0
    rnaalifold_mcc = 0
    centroidalifold_ppvs = []
    centroidalifold_senss = []
    centroidalifold_fprs = []
    centroidalifold_f1_scores = []
    centroidalifold_mccs = []
    petfold_ppvs = []
    petfold_senss = []
    petfold_fprs = []
    petfold_f1_scores = []
    petfold_mccs = []
  gammas = [2. ** i for i in range(min_gamma, max_gamma + 1)]
  conshomfold_ss_dir_path = asset_dir_path + "/conshomfold"
  conshomfold_old_ss_dir_path_turner = asset_dir_path + "/conshomfold_old_turner"
  conshomfold_old_ss_dir_path_contra = asset_dir_path + "/conshomfold_old_contra"
  centroidhomfold_ss_dir_path = asset_dir_path + "/centroidhomfold"
  if False:
    rnaalifold_ss_dir_path = asset_dir_path + "/rnaalifold"
    centroidalifold_ss_dir_path = asset_dir_path + "/centroidalifold"
    petfold_ss_dir_path = asset_dir_path + "/petfold"
  rna_fam_dir_path = asset_dir_path + "/test_ref_sss"
  pool = multiprocessing.Pool(num_of_threads)
  for gamma in gammas:
    conshomfold_count_params = []
    conshomfold_old_count_params_turner = []
    conshomfold_old_count_params_contra = []
    centroidhomfold_count_params = []
    if False:
      rnaalifold_count_params = []
      centroidalifold_count_params = []
      petfold_count_params = []
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
      conshomfold_estimated_ss_file_path = os.path.join(conshomfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      conshomfold_count_params.insert(0, (conshomfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
      conshomfold_old_estimated_ss_dir_path_turner = os.path.join(conshomfold_old_ss_dir_path_turner, rna_fam_name)
      conshomfold_old_estimated_ss_file_path_turner = os.path.join(conshomfold_old_estimated_ss_dir_path_turner, "gamma=" + gamma_str + ".fa")
      conshomfold_old_count_params_turner.insert(0, (conshomfold_old_estimated_ss_file_path_turner, ref_sss, rna_seq_lens))
      conshomfold_old_estimated_ss_dir_path_contra = os.path.join(conshomfold_old_ss_dir_path_contra, rna_fam_name)
      conshomfold_old_estimated_ss_file_path_contra = os.path.join(conshomfold_old_estimated_ss_dir_path_contra, "gamma=" + gamma_str + ".fa")
      conshomfold_old_count_params_contra.insert(0, (conshomfold_old_estimated_ss_file_path_contra, ref_sss, rna_seq_lens))
      centroidhomfold_estimated_ss_dir_path = os.path.join(centroidhomfold_ss_dir_path, rna_fam_name)
      centroidhomfold_estimated_ss_file_path = os.path.join(centroidhomfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      centroidhomfold_count_params.insert(0, (centroidhomfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
      if False:
        if gamma == 1.:
          rnaalifold_estimated_ss_dir_path = os.path.join(rnaalifold_ss_dir_path, )
          rnaalifold_estimated_ss_file_path = os.path.join(rnaalifold_ss_dir_path, rna_fam_name + ".fa")
          rnaalifold_count_params.insert(0, (rnaalifold_estimated_ss_file_path, ref_sss, rna_seq_lens))
        centroidalifold_estimated_ss_dir_path = os.path.join(centroidalifold_ss_dir_path, rna_fam_name)
        centroidalifold_estimated_ss_file_path = os.path.join(centroidalifold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
        centroidalifold_count_params.insert(0, (centroidalifold_estimated_ss_file_path, ref_sss, rna_seq_lens))
        petfold_estimated_ss_dir_path = os.path.join(petfold_ss_dir_path, rna_fam_name)
        petfold_estimated_ss_file_path = os.path.join(petfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
        petfold_count_params.insert(0, (petfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
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
    results = pool.map(get_pos_neg_counts, conshomfold_old_count_params_turner)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    conshomfold_old_ppvs_turner.insert(0, ppv)
    conshomfold_old_senss_turner.insert(0, sens)
    conshomfold_old_fprs_turner.insert(0, fpr)
    conshomfold_old_f1_scores_turner.append(get_f1_score(ppv, sens))
    conshomfold_old_mccs_turner.append(get_mcc(tp, tn, fp, fn))
    results = pool.map(get_pos_neg_counts, conshomfold_old_count_params_contra)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    conshomfold_old_ppvs_contra.insert(0, ppv)
    conshomfold_old_senss_contra.insert(0, sens)
    conshomfold_old_fprs_contra.insert(0, fpr)
    conshomfold_old_f1_scores_contra.append(get_f1_score(ppv, sens))
    conshomfold_old_mccs_contra.append(get_mcc(tp, tn, fp, fn))
    results = pool.map(get_pos_neg_counts, centroidhomfold_count_params)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    centroidhomfold_ppvs.insert(0, ppv)
    centroidhomfold_senss.insert(0, sens)
    centroidhomfold_fprs.insert(0, fpr)
    centroidhomfold_f1_scores.append(get_f1_score(ppv, sens))
    centroidhomfold_mccs.append(get_mcc(tp, tn, fp, fn))
    if False:
      if gamma == 1.:
        results = pool.map(get_pos_neg_counts, rnaalifold_count_params)
        tp, tn, fp, fn = final_sum(results)
        ppv = get_ppv(tp, fp)
        sens = get_sens(tp, fn)
        fpr = get_fpr(tn, fp)
        rnaalifold_ppv = ppv
        rnaalifold_sens = sens
        rnaalifold_fpr = fpr
        rnaalifold_f1_score = get_f1_score(ppv, sens)
        rnaalifold_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_pos_neg_counts, centroidalifold_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      centroidalifold_ppvs.insert(0, ppv)
      centroidalifold_senss.insert(0, sens)
      centroidalifold_fprs.insert(0, fpr)
      centroidalifold_f1_scores.append(get_f1_score(ppv, sens))
      centroidalifold_mccs.append(get_mcc(tp, tn, fp, fn))
      results = pool.map(get_pos_neg_counts, petfold_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      petfold_ppvs.insert(0, ppv)
      petfold_senss.insert(0, sens)
      petfold_fprs.insert(0, fpr)
      petfold_f1_scores.append(get_f1_score(ppv, sens))
      petfold_mccs.append(get_mcc(tp, tn, fp, fn))
  line_1, = pyplot.plot(conshomfold_ppvs, conshomfold_senss, label = "ConsHomfold (new)", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(conshomfold_old_ppvs_turner, conshomfold_old_senss_turner, label = "ConsHomfold (old, Turner)", marker = "s", linestyle = "dashed")
  line_3, = pyplot.plot(conshomfold_old_ppvs_contra, conshomfold_old_senss_contra, label = "ConsHomfold (old, CONTRAfold)", marker = "p", linestyle = "dashdot")
  line_4, = pyplot.plot(centroidhomfold_ppvs, centroidhomfold_senss, label = "CentroidHomfold", marker = "D", linestyle = "dotted")
  if False:
    line_3, = pyplot.plot(rnaalifold_ppv, rnaalifold_sens, label = "RNAalifold", marker = "*", zorder = 10)
    line_4, = pyplot.plot(centroidalifold_ppvs, centroidalifold_senss, label = "CentroidAlifold", marker = "p", linestyle = "dashdot")
    line_5, = pyplot.plot(petfold_ppvs, petfold_senss, label = "PETfold", marker = "D", linestyle = "dotted")
  pyplot.xlabel("Precision")
  pyplot.ylabel("Recall")
  # pyplot.legend(handles = [line_1, line_2, line_3, line_4, line_5], loc = "lower left")
  pyplot.legend(handles = [line_1, line_2, line_3], loc = "lower left")
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/pr_curves_on_ss_estimation_4.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(conshomfold_fprs, conshomfold_senss, label = "ConsHomfold (new)", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(conshomfold_old_fprs_turner, conshomfold_old_senss_turner, label = "ConsHomfold (old, Turner)", marker = "s", linestyle = "dashed")
  line_3, = pyplot.plot(conshomfold_old_fprs_contra, conshomfold_old_senss_contra, label = "ConsHomfold (old, CONTRAfold)", marker = "p", linestyle = "dashdot")
  line_4, = pyplot.plot(centroidhomfold_fprs, centroidhomfold_senss, label = "CentroidHomfold", marker = "D", linestyle = "dotted")
  if False:
    line_3, = pyplot.plot(rnaalifold_fpr, rnaalifold_sens, label = "RNAalifold", marker = "*", zorder = 10)
    line_4, = pyplot.plot(centroidalifold_fprs, centroidalifold_senss, label = "CentroidAlifold", marker = "p", linestyle = "dashdot")
    line_5, = pyplot.plot(petfold_fprs, petfold_senss, label = "PETfold", marker = "D", linestyle = "dotted")
  pyplot.xlabel("Fall-out")
  pyplot.ylabel("Recall")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/roc_curves_on_ss_estimation_4.eps", bbox_inches = "tight")
  pyplot.clf()
  gammas = [i for i in range(min_gamma, max_gamma + 1)]
  line_1, = pyplot.plot(gammas, conshomfold_f1_scores, label = "ConsHomfold (new)", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(gammas, conshomfold_old_f1_scores_turner, label = "ConsHomfold (old, Turner)", marker = "s", linestyle = "dashed")
  line_3, = pyplot.plot(gammas, conshomfold_old_f1_scores_contra, label = "ConsHomfold (old, CONTRAfold)", marker = "P", linestyle = "dashdot")
  line_4, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_f1_scores), max(conshomfold_f1_scores), label = "ConsHomfold (new)", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
  line_5, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_old_f1_scores_turner), max(conshomfold_old_f1_scores_turner), label = "ConsHomfold (old, Turner)", marker = "s", markerfacecolor = white, markeredgecolor = color_palette[1])
  line_6, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_old_f1_scores_contra), max(conshomfold_old_f1_scores_contra), label = "ConsHomfold (old, CONTRAfold)", marker = "p", markerfacecolor = white, markeredgecolor = color_palette[2])
  if False:
    line_3, = pyplot.plot(-2, rnaalifold_f1_score, label = "RNAalifold", marker = "*", zorder = 10)
    line_4, = pyplot.plot(gammas, centroidalifold_f1_scores, label = "CentroidAlifold", marker = "p", linestyle = "dashdot")
    line_5, = pyplot.plot(gammas, petfold_f1_scores, label = "PETfold", marker = "D", linestyle = "dotted")
    line_6, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_f1_scores), max(conshomfold_f1_scores), label = "ConsHomfold (new)", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
    line_7, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_old_f1_scores), max(conshomfold_old_f1_scores), label = "ConsHomfold (old)", marker = "s", markerfacecolor = white, markeredgecolor = color_palette[1])
    line_8, = pyplot.plot(min_gamma + numpy.argmax(centroidalifold_f1_scores), max(centroidalifold_f1_scores), label = "CentroidAlifold", marker = "p", markerfacecolor = white, markeredgecolor = color_palette[3])
    line_9, = pyplot.plot(min_gamma + numpy.argmax(petfold_f1_scores), max(petfold_f1_scores), label = "PETfold", marker = "D", markerfacecolor = white, markeredgecolor = color_palette[4])
    pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("F1 score")
  # pyplot.legend(handles = [line_6, line_7, line_8, line_9], loc = "lower right")
  pyplot.legend(handles = [line_4, line_5, line_6,], loc = "lower right")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_f1_scores_on_ss_estimation_4.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(gammas, conshomfold_mccs, label = "ConsHomfold (new)", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(gammas, conshomfold_old_mccs_turner, label = "ConsHomfold (old, Turner)", marker = "s", linestyle = "dashed")
  line_3, = pyplot.plot(gammas, conshomfold_old_mccs_contra, label = "ConsHomfold (old, CONTRAfold)", marker = "p", linestyle = "dashdot")
  line_4, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_mccs), max(conshomfold_mccs), label = "ConsHomfold (new)", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
  line_5, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_old_mccs_turner), max(conshomfold_old_mccs_turner), label = "ConsHomfold (old, Turner)", marker = "s", markerfacecolor = white, markeredgecolor = color_palette[1])
  line_6, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_old_mccs_contra), max(conshomfold_old_mccs_contra), label = "ConsHomfold (old, CONTRAfold)", marker = "s", markerfacecolor = white, markeredgecolor = color_palette[2])
  if False:
    line_3, = pyplot.plot(-2, rnaalifold_mcc, label = "RNAalifold", marker = "*", zorder = 10)
    line_4, = pyplot.plot(gammas, centroidalifold_mccs, label = "CentroidAlifold", marker = "p", linestyle = "dashdot")
    line_5, = pyplot.plot(gammas, petfold_mccs, label = "PETfold", marker = "D", linestyle = "dotted")
    line_6, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_mccs), max(conshomfold_mccs), label = "ConsHomfold (new)", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
    line_7, = pyplot.plot(min_gamma + numpy.argmax(conshomfold_old_mccs), max(conshomfold_old_mccs), label = "ConsHomfold (old)", marker = "s", markerfacecolor = white, markeredgecolor = color_palette[1])
    line_8, = pyplot.plot(min_gamma + numpy.argmax(centroidalifold_mccs), max(centroidalifold_mccs), label = "CentroidAlifold", marker = "p", markerfacecolor = white, markeredgecolor = color_palette[3])
    line_9, = pyplot.plot(min_gamma + numpy.argmax(petfold_mccs), max(petfold_mccs), label = "PETfold", marker = "D", markerfacecolor = white, markeredgecolor = color_palette[4])
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("MCC")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_mccs_on_ss_estimation_4.eps", bbox_inches = "tight")

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
