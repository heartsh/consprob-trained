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
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  conshomfold_ppvs_trained = []
  conshomfold_senss_trained = []
  conshomfold_fprs_trained = []
  conshomfold_f1_scores_trained = []
  conshomfold_mccs_trained = []
  conshomfold_ppvs = []
  conshomfold_senss = []
  conshomfold_fprs = []
  conshomfold_f1_scores = []
  conshomfold_mccs = []
  centroidhomfold_ppvs = []
  centroidhomfold_senss = []
  centroidhomfold_fprs = []
  centroidhomfold_f1_scores = []
  centroidhomfold_mccs = []
  rnafold_ppv = 0
  rnafold_sens = 0
  rnafold_fpr = 0
  rnafold_f1_score = 0
  rnafold_mcc = 0
  contrafold_ppvs = []
  contrafold_senss = []
  contrafold_fprs = []
  contrafold_f1_scores = []
  contrafold_mccs = []
  centroidfold_ppvs = []
  centroidfold_senss = []
  centroidfold_fprs = []
  centroidfold_f1_scores = []
  centroidfold_mccs = []
  gammas = [2. ** i for i in range(-7, 11)]
  conshomfold_ss_dir_path = asset_dir_path + "/conshomfold"
  conshomfold_ss_dir_path_trained = asset_dir_path + "/conshomfold_trained"
  centroidhomfold_ss_dir_path = asset_dir_path + "/centroidhomfold"
  rnafold_ss_dir_path = asset_dir_path + "/rnafold"
  contrafold_ss_dir_path = asset_dir_path + "/contrafold"
  centroidfold_ss_dir_path = asset_dir_path + "/centroidfold"
  rna_fam_dir_path = asset_dir_path + "/test_ref_sss"
  pool = multiprocessing.Pool(num_of_threads)
  for gamma in gammas:
    conshomfold_count_params = []
    conshomfold_count_params_trained = []
    centroidhomfold_count_params = []
    rnafold_count_params = []
    contrafold_count_params = []
    centroidfold_count_params = []
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
      conshomfold_estimated_ss_dir_path_trained = os.path.join(conshomfold_ss_dir_path_trained, rna_fam_name)
      conshomfold_estimated_ss_file_path = os.path.join(conshomfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".bpseq")
      conshomfold_estimated_ss_file_path_trained = os.path.join(conshomfold_estimated_ss_dir_path_trained, "gamma=" + gamma_str + ".bpseq")
      conshomfold_count_params.insert(0, (conshomfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
      conshomfold_count_params_trained.insert(0, (conshomfold_estimated_ss_file_path_trained, ref_sss, rna_seq_lens))
      centroidhomfold_estimated_ss_dir_path = os.path.join(centroidhomfold_ss_dir_path, rna_fam_name)
      centroidhomfold_estimated_ss_file_path = os.path.join(centroidhomfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      centroidhomfold_count_params.insert(0, (centroidhomfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
      if gamma == 1.:
        rnafold_estimated_ss_dir_path = os.path.join(rnafold_ss_dir_path, )
        rnafold_estimated_ss_file_path = os.path.join(rnafold_ss_dir_path, rna_fam_name + ".fa")
        rnafold_count_params.insert(0, (rnafold_estimated_ss_file_path, ref_sss, rna_seq_lens))
      contrafold_estimated_ss_dir_path = os.path.join(contrafold_ss_dir_path, rna_fam_name)
      contrafold_estimated_ss_file_path = os.path.join(contrafold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      contrafold_count_params.insert(0, (contrafold_estimated_ss_file_path, ref_sss, rna_seq_lens))
      centroidfold_estimated_ss_dir_path = os.path.join(centroidfold_ss_dir_path, rna_fam_name)
      centroidfold_estimated_ss_file_path = os.path.join(centroidfold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
      centroidfold_count_params.insert(0, (centroidfold_estimated_ss_file_path, ref_sss, rna_seq_lens))
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
    results = pool.map(get_pos_neg_counts, conshomfold_count_params_trained)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    conshomfold_ppvs_trained.insert(0, ppv)
    conshomfold_senss_trained.insert(0, sens)
    conshomfold_fprs_trained.insert(0, fpr)
    conshomfold_f1_scores_trained.append(get_f1_score(ppv, sens))
    conshomfold_mccs_trained.append(get_mcc(tp, tn, fp, fn))
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
    if gamma == 1.:
      results = pool.map(get_pos_neg_counts, rnafold_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      rnafold_ppv = ppv
      rnafold_sens = sens
      rnafold_fpr = fpr
      rnafold_f1_score = get_f1_score(ppv, sens)
      rnafold_mcc = get_mcc(tp, tn, fp, fn)
    results = pool.map(get_pos_neg_counts, contrafold_count_params)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    contrafold_ppvs.insert(0, ppv)
    contrafold_senss.insert(0, sens)
    contrafold_fprs.insert(0, fpr)
    contrafold_f1_scores.append(get_f1_score(ppv, sens))
    contrafold_mccs.append(get_mcc(tp, tn, fp, fn))
    results = pool.map(get_pos_neg_counts, centroidfold_count_params)
    tp, tn, fp, fn = final_sum(results)
    ppv = get_ppv(tp, fp)
    sens = get_sens(tp, fn)
    fpr = get_fpr(tn, fp)
    centroidfold_ppvs.insert(0, ppv)
    centroidfold_senss.insert(0, sens)
    centroidfold_fprs.insert(0, fpr)
    centroidfold_f1_scores.append(get_f1_score(ppv, sens))
    centroidfold_mccs.append(get_mcc(tp, tn, fp, fn))
  line_1, = pyplot.plot(conshomfold_ppvs_trained, conshomfold_senss_trained, label = "$ConsHomfold^{train}$", marker = "v", linestyle = "solid")
  line_2, = pyplot.plot(conshomfold_ppvs, conshomfold_senss, label = "$ConsHomfold^{chimera}$", marker = "o", linestyle = "solid")
  line_3, = pyplot.plot(centroidhomfold_ppvs, centroidhomfold_senss, label = "CentroidHomFold", marker = "s", linestyle = "dashed")
  line_4, = pyplot.plot(rnafold_ppv, rnafold_sens, label = "RNAfold", marker = "*")
  line_5, = pyplot.plot(contrafold_ppvs, contrafold_senss, label = "CONTRAfold", marker = "p", linestyle = "dashdot")
  line_6, = pyplot.plot(centroidfold_ppvs, centroidfold_senss, label = "CentroidFold", marker = "D", linestyle = "dotted")
  pyplot.xlabel("Precision")
  pyplot.ylabel("Recall")
  pyplot.legend(handles = [line_1, line_2, line_3, line_4, line_5, line_6], loc = "lower left")
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/pr_curves_on_ss_estimation.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(conshomfold_fprs_trained, conshomfold_senss_trained, label = "$ConsHomfold^{train}$", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(conshomfold_fprs, conshomfold_senss, label = "$ConsHomfold^{train}$", marker = "v", linestyle = "solid")
  line_3, = pyplot.plot(centroidhomfold_fprs, centroidhomfold_senss, label = "CentroidHomfold", marker = "s", linestyle = "dashed")
  line_4, = pyplot.plot(rnafold_fpr, rnafold_sens, label = "RNAfold", marker = "*")
  line_5, = pyplot.plot(contrafold_fprs, contrafold_senss, label = "CONTRAfold", marker = "p", linestyle = "dashdot")
  line_5, = pyplot.plot(centroidfold_fprs, centroidfold_senss, label = "CentroidFold", marker = "D", linestyle = "dotted")
  pyplot.xlabel("Fall-out")
  pyplot.ylabel("Recall")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/roc_curves_on_ss_estimation.eps", bbox_inches = "tight")
  pyplot.clf()
  gammas = [i for i in range(-7, 11)]
  line_1, = pyplot.plot(gammas, conshomfold_f1_scores_trained, label = "$ConsHomfold^{train}", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(gammas, conshomfold_f1_scores, label = "$ConsHomfold^{chimera}$", marker = "v", linestyle = "solid")
  line_3, = pyplot.plot(gammas, centroidhomfold_f1_scores, label = "CentroidHomfold", marker = "s", linestyle = "dashed")
  line_4, = pyplot.plot(-2, rnafold_f1_score, label = "RNAfold", marker = "*")
  line_5, = pyplot.plot(gammas, contrafold_f1_scores, label = "CONTRAfold", marker = "p", linestyle = "dashdot")
  line_6, = pyplot.plot(gammas, centroidfold_f1_scores, label = "CentroidFold", marker = "D", linestyle = "dotted")
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("F1 score")
  pyplot.legend(handles = [line_1, line_2, line_3, line_4, line_5, line_6], loc = "lower right")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_f1_scores_on_ss_estimation.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(gammas, conshomfold_mccs_trained, label = "$ConsHomfold^{train}$", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(gammas, conshomfold_mccs, label = "$ConsHomfold^{chimera}$", marker = "v", linestyle = "solid")
  line_3, = pyplot.plot(gammas, centroidhomfold_mccs, label = "CentroidHomfold", marker = "s", linestyle = "dashed")
  line_4, = pyplot.plot(-2, rnafold_mcc, label = "RNAfold", marker = "*")
  line_5, = pyplot.plot(gammas, contrafold_mccs, label = "CONTRAfold", marker = "p", linestyle = "dashdot")
  line_6, = pyplot.plot(gammas, centroidfold_mccs, label = "CentroidFold", marker = "D", linestyle = "dotted")
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("MCC")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_mccs_on_ss_estimation.eps", bbox_inches = "tight")

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
