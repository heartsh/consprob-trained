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
import glob
from Bio import AlignIO
import pandas
from statistics import mean

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
  consalign_ppv = 0
  consalign_sens = 0
  consalign_fpr = 0
  consalign_f1_score = 0
  consalign_mcc = 0
  consalign_sps = 0
  raf_sens = 0
  raf_ppv = 0
  raf_fpr = 0
  raf_f1_score = 0
  raf_mcc = 0
  raf_sps = 0
  locarna_ppv = 0
  locarna_sens = 0
  locarna_fpr = 0
  locarna_f1_score = 0
  locarna_mcc = 0
  locarna_sps = 0
  dafs_ppv = 0
  dafs_sens = 0
  dafs_fpr = 0
  dafs_f1_score = 0
  dafs_mcc = 0
  dafs_sps = 0
  sparse_ppv = 0
  sparse_sens = 0
  sparse_fpr = 0
  sparse_f1_score = 0
  sparse_mcc = 0
  sparse_sps = 0
  if False:
    turbofold_ppvs = []
    turbofold_senss = []
    turbofold_fprs = []
    turbofold_f1_scores = []
    turbofold_mccs = []
  gammas = [2. ** i for i in range(-7, 11)]
  consalign_ss_dir_path = asset_dir_path + "/consalign"
  raf_ss_dir_path = asset_dir_path + "/raf"
  locarna_ss_dir_path = asset_dir_path + "/locarna"
  dafs_ss_dir_path = asset_dir_path + "/dafs"
  sparse_ss_dir_path = asset_dir_path + "/sparse"
  # turbofold_ss_dir_path = asset_dir_path + "/turbofold"
  rna_fam_dir_path = asset_dir_path + "/test_data"
  ref_sa_dir_path = asset_dir_path + "/test_ref_sas"
  pool = multiprocessing.Pool(num_of_threads)
  for gamma in gammas:
    consalign_count_params = []
    raf_count_params = []
    locarna_count_params = []
    dafs_count_params = []
    sparse_count_params = []
    consalign_count_params_align = []
    raf_count_params_align = []
    locarna_count_params_align = []
    dafs_count_params_align = []
    sparse_count_params_align = []
    # turbofold_count_params = []
    gamma_str = str(gamma) if gamma < 1 else str(int(gamma))
    for rna_fam_file in os.listdir(ref_sa_dir_path):
      if not rna_fam_file.endswith(".sth"):
        continue
      (rna_fam_name, extension) = os.path.splitext(rna_fam_file)
      rna_seq_file_path = os.path.join(rna_fam_dir_path, rna_fam_name + ".fa")
      rna_seq_lens = [len(rna_seq.seq) for rna_seq in SeqIO.parse(rna_seq_file_path, "fasta")]
      # ref_ss_file_path = os.path.join(rna_fam_dir_path, rna_fam_file)
      ref_css_file_path = os.path.join(ref_sa_dir_path, rna_fam_file)
      # ref_sss = utils.get_sss(ref_ss_file_path)
      ref_css = utils.get_css(ref_css_file_path)
      ref_sa = AlignIO.read(ref_css_file_path, "stockholm")
      if gamma == 1.:
        consalign_estimated_ss_dir_path = os.path.join(consalign_ss_dir_path, rna_fam_name)
        os.chdir(consalign_estimated_ss_dir_path)
        for consalign_output_file in glob.glob("consalign.sth"):
          consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path, consalign_output_file)
        # consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
          consalign_count_params.insert(0, (consalign_estimated_ss_file_path, ref_css, rna_seq_lens))
          consalign_count_params_align.insert(0, (consalign_estimated_ss_file_path, ref_sa, rna_seq_lens))
        raf_estimated_ss_file_path = os.path.join(raf_ss_dir_path, rna_fam_file)
        raf_count_params.insert(0, (raf_estimated_ss_file_path, ref_css, rna_seq_lens))
        raf_count_params_align.insert(0, (raf_estimated_ss_file_path, ref_sa, rna_seq_lens))
        locarna_estimated_ss_file_path = os.path.join(locarna_ss_dir_path, rna_fam_file)
        locarna_count_params.insert(0, (locarna_estimated_ss_file_path, ref_css, rna_seq_lens))
        locarna_count_params_align.insert(0, (locarna_estimated_ss_file_path, ref_sa, rna_seq_lens))
        dafs_estimated_ss_file_path = os.path.join(dafs_ss_dir_path, rna_fam_file)
        dafs_count_params.insert(0, (dafs_estimated_ss_file_path, ref_css, rna_seq_lens))
        dafs_count_params_align.insert(0, (dafs_estimated_ss_file_path, ref_sa, rna_seq_lens))
        sparse_estimated_ss_file_path = os.path.join(sparse_ss_dir_path, rna_fam_file)
        sparse_count_params.insert(0, (sparse_estimated_ss_file_path, ref_css, rna_seq_lens))
        sparse_count_params_align.insert(0, (sparse_estimated_ss_file_path, ref_sa, rna_seq_lens))
      if False:
        turbofold_estimated_ss_dir_path = os.path.join(turbofold_ss_dir_path, rna_fam_name)
        turbofold_estimated_ss_file_path = os.path.join(turbofold_estimated_ss_dir_path, "gamma=" + gamma_str + ".fa")
        turbofold_count_params.insert(0, (turbofold_estimated_ss_file_path, ref_sss, rna_seq_lens))
    if gamma == 1.:
      results = pool.map(get_bin_counts, consalign_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      consalign_ppv = ppv
      consalign_sens = sens
      consalign_fpr = fpr
      consalign_f1_score = get_f1_score(ppv, sens)
      consalign_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_bin_counts_align, consalign_count_params_align)
      consalign_sps = get_sps(results)
      results = pool.map(get_bin_counts, raf_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      raf_ppv = ppv
      raf_sens = sens
      raf_fpr = fpr
      raf_f1_score = get_f1_score(ppv, sens)
      raf_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_bin_counts_align, raf_count_params_align)
      raf_sps = get_sps(results)
      results = pool.map(get_bin_counts, locarna_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      locarna_ppv = ppv
      locarna_sens = sens
      locarna_fpr = fpr
      locarna_f1_score = get_f1_score(ppv, sens)
      locarna_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_bin_counts_align, locarna_count_params_align)
      locarna_sps = get_sps(results)
      results = pool.map(get_bin_counts, dafs_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      dafs_ppv = ppv
      dafs_sens = sens
      dafs_fpr = fpr
      dafs_f1_score = get_f1_score(ppv, sens)
      dafs_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_bin_counts_align, dafs_count_params_align)
      dafs_sps = get_sps(results)
      results = pool.map(get_bin_counts, sparse_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      sparse_ppv = ppv
      sparse_sens = sens
      sparse_fpr = fpr
      sparse_f1_score = get_f1_score(ppv, sens)
      sparse_mcc = get_mcc(tp, tn, fp, fn)
      results = pool.map(get_bin_counts_align, sparse_count_params_align)
      sparse_sps = get_sps(results)
    if False:
      results = pool.map(get_bin_counts, turbofold_count_params)
      tp, tn, fp, fn = final_sum(results)
      ppv = get_ppv(tp, fp)
      sens = get_sens(tp, fn)
      fpr = get_fpr(tn, fp)
      turbofold_ppvs.insert(0, ppv)
      turbofold_senss.insert(0, sens)
      turbofold_fprs.insert(0, fpr)
      turbofold_f1_scores.append(get_f1_score(ppv, sens))
      turbofold_mccs.append(get_mcc(tp, tn, fp, fn))
  line_1, = pyplot.plot(consalign_ppv, consalign_sens, label = "ConsHomfold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(raf_ppv, raf_sens, label = "RAF", marker = "s")
  line_3, = pyplot.plot(locarna_ppv, locarna_sens, label = "LocARNA", marker = "*", zorder = 9)
  line_4, = pyplot.plot(dafs_ppv, dafs_sens, label = "DAFS", marker = "p", zorder = 10)
  line_5, = pyplot.plot(sparse_ppv, sparse_sens, label = "SPARSE", marker = "D")
  # line_6, = pyplot.plot(turbofold_ppvs, turbofold_senss, label = "TurboFold", marker = "v", linestyle = "dashed")
  pyplot.xlabel("Precision")
  pyplot.ylabel("Recall")
  # pyplot.legend(handles = [line_1, line_2, line_3, line_4, line_5, line_6], loc = "lower left")
  pyplot.legend(handles = [line_1, line_2, line_3, line_4, line_5], loc = "lower left")
  pyplot.legend(handles = [line_1, line_2], loc = "lower left")
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/pr_curves_on_ss_estimation_3.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(consalign_fpr, consalign_sens, label = "ConsHomfold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(raf_fpr, raf_sens, label = "RAF", marker = "s")
  line_3, = pyplot.plot(locarna_fpr, locarna_sens, label = "LocARNA", marker = "*", zorder = 9)
  line_4, = pyplot.plot(dafs_fpr, dafs_sens, label = "DAFS", marker = "p", zorder = 10)
  line_5, = pyplot.plot(sparse_fpr, sparse_sens, label = "SPARSE", marker = "D")
  # line_6, = pyplot.plot(turbofold_fprs, turbofold_senss, label = "TurboFold", marker = "v", linestyle = "dashed")
  pyplot.xlabel("Fall-out")
  pyplot.ylabel("Recall")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/roc_curves_on_ss_estimation_3.eps", bbox_inches = "tight")
  pyplot.clf()
  gammas = [i for i in range(-7, 11)]
  line_1, = pyplot.plot(-2, consalign_f1_score, label = "ConsHomfold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(-2, raf_f1_score, label = "RAF", marker = "s")
  line_3, = pyplot.plot(-2, locarna_f1_score, label = "LocARNA", marker = "*")
  line_4, = pyplot.plot(-2, dafs_f1_score, label = "DAFS", marker = "p")
  line_5, = pyplot.plot(-2, sparse_f1_score, label = "SPARSE", marker = "D")
  # line_6, = pyplot.plot(gammas, turbofold_f1_scores, label = "TurboFold", marker = "v", linestyle = "dashed")
  # line_7, = pyplot.plot(min_gamma + numpy.argmax(consalign_f1_scores), max(consalign_f1_scores), label = "ConsHomfold", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
  # line_8, = pyplot.plot(min_gamma + numpy.argmax(turbofold_f1_scores), max(turbofold_f1_scores), label = "TurboFold", marker = "v", markerfacecolor = white, markeredgecolor = color_palette[5])
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("F1 score")
  # pyplot.legend(handles = [line_7, line_8], loc = "lower right")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_f1_scores_on_ss_estimation_3.eps", bbox_inches = "tight")
  pyplot.clf()
  line_1, = pyplot.plot(-2, consalign_mcc, label = "ConsHomfold", marker = "o", linestyle = "solid")
  line_2, = pyplot.plot(-2, raf_mcc, label = "RAF", marker = "s")
  line_3, = pyplot.plot(-2, locarna_mcc, label = "LocARNA", marker = "*")
  line_4, = pyplot.plot(-2, dafs_mcc, label = "DAFS", marker = "p")
  line_5, = pyplot.plot(-2, sparse_mcc, label = "SPARSE", marker = "D")
  # line_6, = pyplot.plot(gammas, turbofold_mccs, label = "TurboFold", marker = "v", linestyle = "dashed")
  # line_7, = pyplot.plot(min_gamma + numpy.argmax(consalign_mccs), max(consalign_mccs), label = "ConsHomfold", marker = "o", markerfacecolor = white, markeredgecolor = color_palette[0])
  # line_8, = pyplot.plot(min_gamma + numpy.argmax(turbofold_mccs), max(turbofold_mccs), label = "TurboFold", marker = "v", markerfacecolor = white, markeredgecolor = color_palette[5])
  pyplot.xlabel("$\log_2 \gamma$")
  pyplot.ylabel("MCC")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/gammas_vs_mccs_on_ss_estimation_3.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Sum-of-pairs score": [consalign_sps, raf_sps, locarna_sps, dafs_sps, sparse_sps], "RNA aligner": ["ConsAlign", "RAF", "LocARNA ", "DAFS", "SPARSE"]}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "RNA aligner", y = "Sum-of-pairs score", data = data_frame)
  fig = ax.get_figure()
  fig.tight_layout()
  fig.savefig(image_dir_path + "/rna_aligner_spss.eps", bbox_inches = "tight")
  fig.clf()
  print(consalign_sps)

def get_bin_counts(params):
  # rna_seq_lens, estimated_css, ref_css = params
  (estimated_ss_file_path, ref_css, rna_seq_lens) = params
  num_of_rnas = len(rna_seq_lens)
  estimated_css = utils.get_css(estimated_ss_file_path)
  tp = fp = tn = fn = 0
  for m in range(0, num_of_rnas):
    sub_estimated_css = estimated_css[m]
    sub_ref_css = ref_css[m]
    rna_seq_len_1 = rna_seq_lens[m]
    for i in range(0, rna_seq_len_1):
      for j in range(i + 1, rna_seq_len_1):
        estimated_bin = (i, j) in sub_estimated_css
        ref_bin = (i, j) in sub_ref_css
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

def get_bin_counts_align(params):
  (estimated_sa_file_path, ref_sa, rna_seq_lens) = params
  num_of_rnas = len(rna_seq_lens)
  estimated_sa = AlignIO.read(estimated_sa_file_path, "stockholm")
  tp = total = 0
  estimated_sa_len = len(estimated_sa[0])
  num_of_rnas = len(estimated_sa)
  pos_map_sets_estimated = []
  for i in range(num_of_rnas):
    pos_maps_estimated = []
    pos = -1
    for j in range(estimated_sa_len):
      c = estimated_sa[i][j]
      if c != "-":
        pos += 1
      if c != "-":
        pos_maps_estimated.append(pos)
      else:
        pos_maps_estimated.append(-1)
    pos_map_sets_estimated.append(pos_maps_estimated)
  aligned_pair_sets_estimated = {}
  for m in range(0, num_of_rnas):
    pos_maps_estimated = pos_map_sets_estimated[m]
    for n in range(m + 1, num_of_rnas):
      aligned_pairs_estimated = set()
      pos_maps_estimated_2 = pos_map_sets_estimated[n]
      for i in range(estimated_sa_len):
        pos_pair = (pos_maps_estimated[i], pos_maps_estimated_2[i])
        if pos_pair[0] != -1 or pos_pair[1] != -1:
          aligned_pairs_estimated.add(pos_pair)
      aligned_pair_sets_estimated[(m, n)] = aligned_pairs_estimated
  ref_sa_len = len(ref_sa[0])
  ref_pos_map_sets = []
  for i in range(num_of_rnas):
    ref_pos_maps = []
    pos = -1
    for j in range(ref_sa_len):
      c = ref_sa[i][j]
      if c != "-":
        pos += 1
      if c != "-":
        ref_pos_maps.append(pos)
      else:
        ref_pos_maps.append(-1)
    ref_pos_map_sets.append(ref_pos_maps)
  for m in range(0, num_of_rnas):
    ref_pos_maps = ref_pos_map_sets[m]
    for n in range(m + 1, num_of_rnas):
      ref_pos_maps_2 = ref_pos_map_sets[n]
      aligned_pos_pairs_estimated = aligned_pair_sets_estimated[(m, n)]
      for i in range(ref_sa_len):
        pos_pair = (ref_pos_maps[i], ref_pos_maps_2[i])
        if pos_pair[0] != -1 or pos_pair[1] != -1:
          if pos_pair in aligned_pos_pairs_estimated:
            tp += 1
          total += 1
  return tp, total

def final_sum(results):
  final_tp = final_tn = final_fp = final_fn = 0.
  for tp, tn, fp, fn in results:
    final_tp += tp
    final_tn += tn
    final_fp += fp
    final_fn += fn
  return (final_tp, final_tn, final_fp, final_fn)

def get_sps(results):
  return mean(list(map(lambda x: x[0] / x[1], results)))
  if False:
    final_tp = final_total = 0.
    for tp, total in results:
      final_tp += tp
      final_total += total
    return final_tp / final_total

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
