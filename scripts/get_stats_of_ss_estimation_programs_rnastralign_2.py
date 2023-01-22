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

seaborn.set(font_scale = 1.2)
color_palette = seaborn.color_palette()
color_palette_2 = seaborn.color_palette("Set2")
white = "#F2F2F2"
pyplot.rcParams["figure.dpi"] = 600

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  consalign_ss_dir_path = asset_dir_path + "/consalign_rnastralign"
  rna_dir_path = asset_dir_path + "/RNAStrAlign_sampled"
  pool = multiprocessing.Pool(num_of_threads)
  consalign_count_params = []
  consalign_count_params_align = []
  for rna_sub_dir in os.listdir(rna_dir_path):
    rna_sub_dir_path = os.path.join(rna_dir_path, rna_sub_dir)
    rna_file_path = os.path.join(rna_sub_dir_path, "sampled_seqs.fa")
    if not os.path.isfile(rna_file_path):
      continue
    rna_seq_lens = [len(rna_seq.seq) for rna_seq in SeqIO.parse(rna_file_path, "fasta")]
    seq_num = len(rna_seq_lens)
    ref_sss = utils.get_sss_ct(rna_sub_dir_path, seq_num)
    ref_sa_file_path = os.path.join(rna_sub_dir_path, "%s.aln" % rna_sub_dir)
    ref_sa = AlignIO.read(ref_sa_file_path, "clustal")
    consalign_estimated_ss_dir_path = os.path.join(consalign_ss_dir_path, rna_sub_dir)
    os.chdir(consalign_estimated_ss_dir_path)
    for consalign_output_file in glob.glob("consalign.sth"):
      consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path, consalign_output_file)
      consalign_count_params_align.insert(0, (consalign_estimated_ss_file_path, ref_sa, rna_seq_lens))
  consalign_metrics = list(pool.map(get_metrics, consalign_count_params_align))
  consalign_spss_empirical = list(map(lambda x: x[0], consalign_metrics))
  consalign_spss_expected = list(map(lambda x: x[1], consalign_metrics))
  consalign_gammas_bp = list(map(lambda x: x[2], consalign_metrics))
  consalign_gammas_align = list(map(lambda x: x[3], consalign_metrics))
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  data = {"Empirical sum-of-pairs score": consalign_spss_empirical, "Expected sum-of-pairs score": consalign_spss_expected}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.jointplot(x = "Empirical sum-of-pairs score", y = "Expected sum-of-pairs score", data = data_frame, kind = "kde", fill = True)
  ax.plot_joint(seaborn.regplot, scatter = False, color = color_palette[1])
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/consalign_sps_corr_rnastralign.svg", bbox_inches = "tight")
  pyplot.clf()
  print(data_frame.corr())
  data = {r"$\gamma^{\rm M}$": consalign_gammas_align, r"$\gamma^{\rm P}$": consalign_gammas_bp}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.jointplot(x = r"$\gamma^{\rm M}$", y = r"$\gamma^{\rm P}$", data = data_frame, kind = "kde", fill = True)
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/consalign_gamma_corr_rnastralign.svg", bbox_inches = "tight")
  pyplot.clf()

def get_bin_counts(params):
  (estimated_ss_file_path, ref_css, rna_seq_lens) = params
  num_of_rnas = len(rna_seq_lens)
  estimated_css = utils.get_css(estimated_ss_file_path) if estimated_ss_file_path.endswith(".sth") else utils.get_sss(estimated_ss_file_path)
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

def get_metrics(params):
  (estimated_sa_file_path, ref_sa, rna_seq_lens) = params
  num_of_rnas = len(rna_seq_lens)
  estimated_sa = AlignIO.read(estimated_sa_file_path, "stockholm" if estimated_sa_file_path.endswith(".sth") else "fasta")
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
          total += 1
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
  f = open(estimated_sa_file_path, "r")
  line = f.readlines()[1]
  splits = line.split()
  string = splits[-1]
  expected_sps = float(string.split("=")[-1])
  string = splits[-2]
  gamma_bp = float(string.split("=")[-1])
  string = splits[-3]
  gamma_align = float(string.split("=")[-1])
  return (tp / total, expected_sps, gamma_bp, gamma_align)

def get_sci(params):
  (estimated_sa_file_path, ref_sa, rna_seq_lens) = params
  rnaalifold_command = "RNAalifold %s --sci" % estimated_sa_file_path
  (output, _, _) = utils.run_command(rnaalifold_command)
  output = str(output)
  splits = output.split("sci = ")
  sci = splits[-1].strip()
  sci = float(sci[:-4])
  return sci

def get_f1_score(result):
  (tp, tn, fp, fn) = result
  denom = tp + 0.5 * (fp + fn)
  return tp / denom

def get_mcc(result):
  (tp, tn, fp, fn) = result
  denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  return (tp * tn - fp * fn) / denom

if __name__ == "__main__":
  main()
