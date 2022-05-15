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

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  consalign_ss_dir_path = asset_dir_path + "/consalign_rnastralign"
  consalign_ss_dir_path_disabled_alifold = asset_dir_path + "/consalign_rnastralign_disabled_alifold"
  raf_ss_dir_path = asset_dir_path + "/raf_rnastralign"
  locarna_ss_dir_path = asset_dir_path + "/locarna_rnastralign"
  dafs_ss_dir_path = asset_dir_path + "/dafs_rnastralign"
  sparse_ss_dir_path = asset_dir_path + "/sparse_rnastralign"
  turbofold_ss_dir_path = asset_dir_path + "/turbofold_rnastralign"
  rna_dir_path = asset_dir_path + "/RNAStrAlign_sampled"
  pool = multiprocessing.Pool(num_of_threads)
  consalign_count_params = []
  consalign_count_params_disabled_alifold = []
  raf_count_params = []
  locarna_count_params = []
  dafs_count_params = []
  sparse_count_params = []
  turbofold_count_params = []
  consalign_count_params_align = []
  consalign_count_params_align_disabled_alifold = []
  raf_count_params_align = []
  locarna_count_params_align = []
  dafs_count_params_align = []
  sparse_count_params_align = []
  turbofold_count_params_align = []
  for rna_sub_dir in os.listdir(rna_dir_path):
    rna_sub_dir_path = os.path.join(rna_dir_path, rna_sub_dir)
    rna_file_path = os.path.join(rna_sub_dir_path, "sampled_seqs.fa")
    if not os.path.isfile(rna_file_path):
      continue
    rna_seq_lens = [len(rna_seq.seq) for rna_seq in SeqIO.parse(rna_file_path, "fasta")]
    ref_sss = utils.get_sss_ct(rna_sub_dir_path)
    ref_sa_file_path = os.path.join(rna_sub_dir_path, "%s.aln" % rna_sub_dir)
    ref_sa = AlignIO.read(ref_sa_file_path, "clustal")
    consalign_estimated_ss_dir_path = os.path.join(consalign_ss_dir_path, rna_sub_dir)
    os.chdir(consalign_estimated_ss_dir_path)
    for consalign_output_file in glob.glob("consalign.sth"):
      consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path, consalign_output_file)
      consalign_count_params.insert(0, (consalign_estimated_ss_file_path, ref_sss, rna_seq_lens))
      consalign_count_params_align.insert(0, (consalign_estimated_ss_file_path, ref_sa, rna_seq_lens))
    consalign_estimated_ss_dir_path_disabled_alifold = os.path.join(consalign_ss_dir_path_disabled_alifold, rna_sub_dir)
    os.chdir(consalign_estimated_ss_dir_path_disabled_alifold)
    for consalign_output_file in glob.glob("consalign.sth"):
      consalign_estimated_ss_file_path_disabled_alifold = os.path.join(consalign_estimated_ss_dir_path_disabled_alifold, consalign_output_file)
      consalign_count_params_disabled_alifold.insert(0, (consalign_estimated_ss_file_path_disabled_alifold, ref_sss, rna_seq_lens))
      consalign_count_params_align_disabled_alifold.insert(0, (consalign_estimated_ss_file_path_disabled_alifold, ref_sa, rna_seq_lens))
    raf_estimated_ss_file_path = os.path.join(raf_ss_dir_path, "%s/%s.sth" % (rna_sub_dir, rna_sub_dir))
    raf_count_params.insert(0, (raf_estimated_ss_file_path, ref_sss, rna_seq_lens))
    raf_count_params_align.insert(0, (raf_estimated_ss_file_path, ref_sa, rna_seq_lens))
    locarna_estimated_ss_file_path = os.path.join(locarna_ss_dir_path, "%s/%s.sth" % (rna_sub_dir, rna_sub_dir))
    locarna_count_params.insert(0, (locarna_estimated_ss_file_path, ref_sss, rna_seq_lens))
    locarna_count_params_align.insert(0, (locarna_estimated_ss_file_path, ref_sa, rna_seq_lens))
    dafs_estimated_ss_file_path = os.path.join(dafs_ss_dir_path, "%s/%s.sth" % (rna_sub_dir, rna_sub_dir))
    dafs_count_params.insert(0, (dafs_estimated_ss_file_path, ref_sss, rna_seq_lens))
    dafs_count_params_align.insert(0, (dafs_estimated_ss_file_path, ref_sa, rna_seq_lens))
    sparse_estimated_ss_file_path = os.path.join(sparse_ss_dir_path, "%s/%s.sth" % (rna_sub_dir, rna_sub_dir))
    sparse_count_params.insert(0, (sparse_estimated_ss_file_path, ref_sss, rna_seq_lens))
    sparse_count_params_align.insert(0, (sparse_estimated_ss_file_path, ref_sa, rna_seq_lens))
    turbofold_estimated_ss_dir_path = os.path.join(turbofold_ss_dir_path, rna_sub_dir)
    turbofold_estimated_ss_file_path = os.path.join(turbofold_estimated_ss_dir_path, "output.fa")
    turbofold_count_params.insert(0, (turbofold_estimated_ss_file_path, ref_sss, rna_seq_lens))
    turbofold_estimated_sa_file_path = os.path.join(turbofold_estimated_ss_dir_path, "output.aln")
    turbofold_count_params_align.insert(0, (turbofold_estimated_sa_file_path, ref_sa, rna_seq_lens))
  results = numpy.sum(pool.map(get_bin_counts, consalign_count_params), axis = 0).tolist()
  consalign_f1_score = get_f1_score(results)
  consalign_mcc = get_mcc(results)
  consalign_spss = pool.map(get_sps, consalign_count_params_align)
  consalign_scis = pool.map(get_sci, consalign_count_params_align)
  consalign_sps = numpy.mean(consalign_spss)
  consalign_sci = numpy.mean(consalign_scis)
  results = numpy.sum(pool.map(get_bin_counts, consalign_count_params_disabled_alifold), axis = 0).tolist()
  consalign_f1_score_disabled_alifold = get_f1_score(results)
  consalign_mcc_disabled_alifold = get_mcc(results)
  results = numpy.sum(pool.map(get_bin_counts, raf_count_params), axis = 0).tolist()
  raf_f1_score = get_f1_score(results)
  raf_mcc = get_mcc(results)
  raf_spss = pool.map(get_sps, raf_count_params_align)
  raf_scis = pool.map(get_sci, raf_count_params_align)
  raf_sps = numpy.mean(raf_spss)
  raf_sci = numpy.mean(raf_scis)
  results = numpy.sum(pool.map(get_bin_counts, locarna_count_params), axis = 0).tolist()
  locarna_f1_score = get_f1_score(results)
  locarna_mcc = get_mcc(results)
  locarna_spss = pool.map(get_sps, locarna_count_params_align)
  locarna_scis = pool.map(get_sci, locarna_count_params_align)
  locarna_sps = numpy.mean(locarna_spss)
  locarna_sci = numpy.mean(locarna_scis)
  results = numpy.sum(pool.map(get_bin_counts, dafs_count_params), axis = 0).tolist()
  dafs_f1_score = get_f1_score(results)
  dafs_mcc = get_mcc(results)
  dafs_spss = pool.map(get_sps, dafs_count_params_align)
  dafs_scis = pool.map(get_sci, dafs_count_params_align)
  dafs_sps = numpy.mean(dafs_spss)
  dafs_sci = numpy.mean(dafs_scis)
  results = numpy.sum(pool.map(get_bin_counts, sparse_count_params), axis = 0).tolist()
  sparse_f1_score = get_f1_score(results)
  sparse_mcc = get_mcc(results)
  sparse_spss = pool.map(get_sps, sparse_count_params_align)
  sparse_scis = pool.map(get_sci, sparse_count_params_align)
  sparse_sps = numpy.mean(sparse_spss)
  sparse_sci = numpy.mean(sparse_scis)
  results = numpy.sum(pool.map(get_bin_counts, turbofold_count_params), axis = 0).tolist()
  turbofold_f1_score = get_f1_score(results)
  turbofold_mcc = get_mcc(results)
  turbofold_spss = pool.map(get_sps, turbofold_count_params_align)
  turbofold_scis = pool.map(get_sci, turbofold_count_params_align)
  turbofold_sps = numpy.mean(turbofold_spss)
  turbofold_sci = numpy.mean(turbofold_scis)
  data = {"Structure prediction accuracy": [consalign_f1_score, raf_f1_score, locarna_f1_score, dafs_f1_score, sparse_f1_score, turbofold_f1_score] + [consalign_mcc, raf_mcc, locarna_mcc, dafs_mcc, sparse_mcc, turbofold_mcc], "RNA structural aligner": ["ConsAlign", "RAF", "LocARNA ", "DAFS", "SPARSE", "Linear\nTurboFold"] * 2, "Structure accuracy type": ["F1 score"] * 6 + ["Matthews correlation coefficient"] * 6}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "RNA structural aligner", y = "Structure prediction accuracy", data = data_frame, hue = "Structure accuracy type")
  ax.set(ylim = (0, 0.5))
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/rna_aligner_rnastralign.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Sum-of-pairs score": [consalign_sps, raf_sps, locarna_sps, dafs_sps, sparse_sps, turbofold_sps], "RNA structural aligner": ["ConsAlign", "RAF", "LocARNA ", "DAFS", "SPARSE", "Linear\nTurboFold"]}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.scatterplot(x = "RNA structural aligner", y = "Sum-of-pairs score", data = data_frame, zorder = 10, marker = "+", color = "white", s = 100)
  data = {"Sum-of-pairs score": consalign_spss + raf_spss + locarna_spss + dafs_spss + sparse_spss + turbofold_spss, "RNA structural aligner": ["ConsAlign"] * len(consalign_spss) + ["RAF"] * len(raf_spss) + ["LocARNA"] * len(locarna_spss) + ["DAFS"] * len(dafs_spss) + ["SPARSE"] * len(sparse_spss) + ["Linear\nTurboFold"] * len(turbofold_spss)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "RNA structural aligner", y = "Sum-of-pairs score", data = data_frame, sym  = "")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/rna_aligner_spss_box_plot_rnastralign.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Structure conservation index": [consalign_sci, raf_sci, locarna_sci, dafs_sci, sparse_sci, turbofold_sci], "RNA structural aligner": ["ConsAlign", "RAF", "LocARNA ", "DAFS", "SPARSE", "Linear\nTurboFold"]}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.scatterplot(x = "RNA structural aligner", y = "Structure conservation index", data = data_frame, zorder = 10, marker = "+", color = "white", s = 100)
  data = {"Structure conservation index": consalign_scis + raf_scis + locarna_scis + dafs_scis + sparse_scis + turbofold_scis, "RNA structural aligner": ["ConsAlign"] * len(consalign_scis) + ["RAF"] * len(raf_scis) + ["LocARNA"] * len(locarna_scis) + ["DAFS"] * len(dafs_scis) + ["SPARSE"] * len(sparse_scis) + ["Linear\nTurboFold"] * len(turbofold_scis)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "RNA structural aligner", y = "Structure conservation index", data = data_frame, sym  = "")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/rna_aligner_scis_box_plot_rnastralign.eps", bbox_inches = "tight")
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

def get_sps(params):
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
  sps = tp / total
  return sps

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
