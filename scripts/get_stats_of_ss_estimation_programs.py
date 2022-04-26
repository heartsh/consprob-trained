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
pyplot.rcParams['legend.handlelength'] = 0
pyplot.rcParams['legend.fontsize'] = "x-large"
color_palette = seaborn.color_palette()
white = "#F2F2F2"

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  consalign_ss_dir_path_ensemble = asset_dir_path + "/consalign_ensemble"
  consalign_ss_dir_path_turner = asset_dir_path + "/consalign_turner"
  consalign_ss_dir_path_trained = asset_dir_path + "/consalign_trained_transfer"
  consalign_ss_dir_path_trained_random_init = asset_dir_path + "/consalign_trained_random_init"
  consalign_ss_dir_path_transferred_only = asset_dir_path + "/consalign_transferred_only"
  raf_ss_dir_path = asset_dir_path + "/raf"
  locarna_ss_dir_path = asset_dir_path + "/locarna"
  dafs_ss_dir_path = asset_dir_path + "/dafs"
  sparse_ss_dir_path = asset_dir_path + "/sparse"
  turbofold_ss_dir_path = asset_dir_path + "/turbofold"
  rna_fam_dir_path = asset_dir_path + "/test_data"
  ref_sa_dir_path = asset_dir_path + "/test_ref_sas"
  pool = multiprocessing.Pool(num_of_threads)
  consalign_count_params_ensemble = []
  consalign_count_params_turner = []
  consalign_count_params_trained = []
  consalign_count_params_trained_random_init = []
  consalign_count_params_transferred_only = []
  raf_count_params = []
  locarna_count_params = []
  dafs_count_params = []
  sparse_count_params = []
  consalign_count_params_align_ensemble = []
  consalign_count_params_align_turner = []
  consalign_count_params_align_trained = []
  consalign_count_params_align_trained_random_init = []
  consalign_count_params_align_transferred_only = []
  raf_count_params_align = []
  locarna_count_params_align = []
  dafs_count_params_align = []
  sparse_count_params_align = []
  turbofold_count_params = []
  turbofold_count_params_align = []
  for rna_fam_file in os.listdir(ref_sa_dir_path):
    if not rna_fam_file.endswith(".sth"):
      continue
    (rna_fam_name, extension) = os.path.splitext(rna_fam_file)
    rna_seq_file_path = os.path.join(rna_fam_dir_path, rna_fam_name + ".fa")
    rna_seq_lens = [len(rna_seq.seq) for rna_seq in SeqIO.parse(rna_seq_file_path, "fasta")]
    ref_css_file_path = os.path.join(ref_sa_dir_path, rna_fam_file)
    ref_css = utils.get_css(ref_css_file_path)
    ref_sa = AlignIO.read(ref_css_file_path, "stockholm")
    consalign_estimated_ss_dir_path_ensemble = os.path.join(consalign_ss_dir_path_ensemble, rna_fam_name)
    os.chdir(consalign_estimated_ss_dir_path_ensemble)
    for consalign_output_file in glob.glob("consalign.sth"):
      consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path_ensemble, consalign_output_file)
      consalign_count_params_ensemble.insert(0, (consalign_estimated_ss_file_path, ref_css, rna_seq_lens))
      consalign_count_params_align_ensemble.insert(0, (consalign_estimated_ss_file_path, ref_sa, rna_seq_lens))
    consalign_estimated_ss_dir_path_turner = os.path.join(consalign_ss_dir_path_turner, rna_fam_name)
    os.chdir(consalign_estimated_ss_dir_path_turner)
    for consalign_output_file in glob.glob("consalign.sth"):
      consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path_turner, consalign_output_file)
      consalign_count_params_turner.insert(0, (consalign_estimated_ss_file_path, ref_css, rna_seq_lens))
      consalign_count_params_align_turner.insert(0, (consalign_estimated_ss_file_path, ref_sa, rna_seq_lens))
    consalign_estimated_ss_dir_path_trained = os.path.join(consalign_ss_dir_path_trained, rna_fam_name)
    os.chdir(consalign_estimated_ss_dir_path_trained)
    for consalign_output_file in glob.glob("consalign.sth"):
      consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path_trained, consalign_output_file)
      consalign_count_params_trained.insert(0, (consalign_estimated_ss_file_path, ref_css, rna_seq_lens))
      consalign_count_params_align_trained.insert(0, (consalign_estimated_ss_file_path, ref_sa, rna_seq_lens))
    consalign_estimated_ss_dir_path_trained_random_init = os.path.join(consalign_ss_dir_path_trained_random_init, rna_fam_name)
    os.chdir(consalign_estimated_ss_dir_path_trained_random_init)
    for consalign_output_file in glob.glob("consalign.sth"):
      consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path_trained_random_init, consalign_output_file)
      consalign_count_params_trained_random_init.insert(0, (consalign_estimated_ss_file_path, ref_css, rna_seq_lens))
      consalign_count_params_align_trained_random_init.insert(0, (consalign_estimated_ss_file_path, ref_sa, rna_seq_lens))
    consalign_estimated_ss_dir_path_transferred_only = os.path.join(consalign_ss_dir_path_transferred_only, rna_fam_name)
    os.chdir(consalign_estimated_ss_dir_path_transferred_only)
    for consalign_output_file in glob.glob("consalign.sth"):
      consalign_estimated_ss_file_path = os.path.join(consalign_estimated_ss_dir_path_transferred_only, consalign_output_file)
      consalign_count_params_transferred_only.insert(0, (consalign_estimated_ss_file_path, ref_css, rna_seq_lens))
      consalign_count_params_align_transferred_only.insert(0, (consalign_estimated_ss_file_path, ref_sa, rna_seq_lens))
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
    turbofold_estimated_ss_dir_path = os.path.join(turbofold_ss_dir_path, rna_fam_name)
    turbofold_estimated_ss_file_path = os.path.join(turbofold_estimated_ss_dir_path, "output.fa")
    turbofold_count_params.insert(0, (turbofold_estimated_ss_file_path, ref_css, rna_seq_lens))
    turbofold_estimated_sa_file_path = os.path.join(turbofold_estimated_ss_dir_path, "output.aln")
    turbofold_count_params_align.insert(0, (turbofold_estimated_sa_file_path, ref_sa, rna_seq_lens))
  results = pool.map(get_bin_counts, consalign_count_params_ensemble)
  consalign_f1_scores_ensemble = pool.map(get_f1_score, results)
  consalign_f1_score_ensemble = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_f1_scores_ensemble)))
  consalign_mccs_ensemble = pool.map(get_mcc, results)
  consalign_mcc_ensemble = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_mccs_ensemble)))
  consalign_spss_ensemble = pool.map(get_sps, consalign_count_params_align_ensemble)
  consalign_sps_ensemble = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_spss_ensemble)))
  results = pool.map(get_bin_counts, consalign_count_params_turner)
  consalign_f1_scores_turner = pool.map(get_f1_score, results)
  consalign_f1_score_turner = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_f1_scores_turner)))
  consalign_mccs_turner = pool.map(get_mcc, results)
  consalign_mcc_turner = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_mccs_turner)))
  consalign_spss_turner = pool.map(get_sps, consalign_count_params_align_turner)
  consalign_sps_turner = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_spss_turner)))
  results = pool.map(get_bin_counts, consalign_count_params_trained)
  consalign_f1_scores_trained = pool.map(get_f1_score, results)
  consalign_f1_score_trained = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_f1_scores_trained)))
  consalign_mccs_trained = pool.map(get_mcc, results)
  consalign_mcc_trained = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_mccs_trained)))
  consalign_spss_trained = pool.map(get_sps, consalign_count_params_align_trained)
  consalign_sps_trained = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_spss_trained)))
  results = pool.map(get_bin_counts, consalign_count_params_trained_random_init)
  consalign_f1_scores_trained_random_init = pool.map(get_f1_score, results)
  consalign_f1_score_trained_random_init = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_f1_scores_trained_random_init)))
  consalign_mccs_trained_random_init = pool.map(get_mcc, results)
  consalign_mcc_trained_random_init = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_mccs_trained_random_init)))
  consalign_spss_trained_random_init = pool.map(get_sps, consalign_count_params_align_trained_random_init)
  consalign_sps_trained_random_init = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_spss_trained_random_init)))
  results = pool.map(get_bin_counts, consalign_count_params_transferred_only)
  consalign_f1_scores_transferred_only = pool.map(get_f1_score, results)
  consalign_f1_score_transferred_only = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_f1_scores_transferred_only)))
  consalign_mccs_transferred_only = pool.map(get_mcc, results)
  consalign_mcc_transferred_only = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_mccs_transferred_only)))
  consalign_spss_transferred_only = pool.map(get_sps, consalign_count_params_align_transferred_only)
  consalign_sps_transferred_only = numpy.mean(list(filter(lambda x: x > float('-inf'), consalign_spss_transferred_only)))
  results = pool.map(get_bin_counts, raf_count_params)
  raf_f1_scores = pool.map(get_f1_score, results)
  raf_f1_score = numpy.mean(list(filter(lambda x: x > float('-inf'), raf_f1_scores)))
  raf_mccs = pool.map(get_mcc, results)
  raf_mcc = numpy.mean(list(filter(lambda x: x > float('-inf'), raf_mccs)))
  raf_spss = pool.map(get_sps, raf_count_params_align)
  raf_sps = numpy.mean(list(filter(lambda x: x > float('-inf'), raf_spss)))
  results = pool.map(get_bin_counts, locarna_count_params)
  locarna_f1_scores = pool.map(get_f1_score, results)
  locarna_f1_score = numpy.mean(list(filter(lambda x: x > float('-inf'), locarna_f1_scores)))
  locarna_mccs = pool.map(get_mcc, results)
  locarna_mcc = numpy.mean(list(filter(lambda x: x > float('-inf'), locarna_mccs)))
  locarna_spss = pool.map(get_sps, locarna_count_params_align)
  locarna_sps = numpy.mean(list(filter(lambda x: x > float('-inf'), locarna_spss)))
  results = pool.map(get_bin_counts, dafs_count_params)
  dafs_f1_scores = pool.map(get_f1_score, results)
  dafs_f1_score = numpy.mean(list(filter(lambda x: x > float('-inf'), dafs_f1_scores)))
  dafs_mccs = pool.map(get_mcc, results)
  dafs_mcc = numpy.mean(list(filter(lambda x: x > float('-inf'), dafs_mccs)))
  dafs_spss = pool.map(get_sps, dafs_count_params_align)
  dafs_sps = numpy.mean(list(filter(lambda x: x > float('-inf'), dafs_spss)))
  results = pool.map(get_bin_counts, sparse_count_params)
  sparse_f1_scores = pool.map(get_f1_score, results)
  sparse_f1_score = numpy.mean(list(filter(lambda x: x > float('-inf'), sparse_f1_scores)))
  sparse_mccs = pool.map(get_mcc, results)
  sparse_mcc = numpy.mean(list(filter(lambda x: x > float('-inf'), sparse_mccs)))
  sparse_spss = pool.map(get_sps, sparse_count_params_align)
  sparse_sps = numpy.mean(list(filter(lambda x: x > float('-inf'), sparse_spss)))
  results = pool.map(get_bin_counts, turbofold_count_params)
  turbofold_f1_scores = pool.map(get_f1_score, results)
  turbofold_f1_score = numpy.mean(list(filter(lambda x: x > float('-inf'), turbofold_f1_scores)))
  turbofold_mccs = pool.map(get_mcc, results)
  turbofold_mcc = numpy.mean(list(filter(lambda x: x > float('-inf'), turbofold_mccs)))
  turbofold_spss = pool.map(get_sps, turbofold_count_params_align)
  turbofold_sps = numpy.mean(list(filter(lambda x: x > float('-inf'), turbofold_spss)))
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  data = {"F1 score": consalign_f1_scores_ensemble + raf_f1_scores + locarna_f1_scores + dafs_f1_scores + sparse_f1_scores + turbofold_f1_scores, "RNA structural aligner": ["ConsAlign"] * len(consalign_f1_scores_ensemble) + ["RAF"] * len(raf_f1_scores) + ["LocARNA"] * len(locarna_f1_scores) + ["DAFS"] * len(dafs_f1_scores) + ["SPARSE"] * len(sparse_f1_scores) + ["Linear\nTurboFold"] * len(turbofold_f1_scores)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "RNA structural aligner", y = "F1 score", data = data_frame)
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/rna_aligner_f1_scores_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Matthews correlation coefficient": consalign_mccs_ensemble + raf_mccs + locarna_mccs + dafs_mccs + sparse_mccs + turbofold_mccs, "RNA structural aligner": ["ConsAlign"] * len(consalign_mccs_ensemble) + ["RAF"] * len(raf_mccs) + ["LocARNA"] * len(locarna_mccs) + ["DAFS"] * len(dafs_mccs) + ["SPARSE"] * len(sparse_mccs) + ["Linear\nTurboFold"] * len(turbofold_mccs)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "RNA structural aligner", y = "Matthews correlation coefficient", data = data_frame)
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/rna_aligner_mccs_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Sum-of-pairs score": consalign_spss_ensemble + raf_spss + locarna_spss + dafs_spss + sparse_spss + turbofold_spss, "RNA structural aligner": ["ConsAlign"] * len(consalign_spss_ensemble) + ["RAF"] * len(raf_spss) + ["LocARNA"] * len(locarna_spss) + ["DAFS"] * len(dafs_spss) + ["SPARSE"] * len(sparse_spss) + ["Linear\nTurboFold"] * len(turbofold_spss)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "RNA structural aligner", y = "Sum-of-pairs score", data = data_frame, sym = "")
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/rna_aligner_spss_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"F1 score": consalign_f1_scores_ensemble + consalign_f1_scores_turner + consalign_f1_scores_trained, "Structural alignment scoring model": ["Ensemble"] * len(consalign_f1_scores_ensemble) + ["Turner"] * len(consalign_f1_scores_turner) + ["Trained"] * len(consalign_f1_scores_trained)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "Structural alignment scoring model", y = "F1 score", data = data_frame)
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/consalign_model_comparison_f1_score_scoring_model_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Matthews correlation coefficient": consalign_mccs_ensemble + consalign_mccs_turner + consalign_mccs_trained, "Structural alignment scoring model": ["Ensemble"] * len(consalign_mccs_ensemble) + ["Turner"] * len(consalign_mccs_turner) + ["Trained"] * len(consalign_mccs_trained)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "Structural alignment scoring model", y = "Matthews correlation coefficient", data = data_frame)
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/consalign_model_comparison_mcc_scoring_model_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Sum-of-pairs score": consalign_spss_ensemble + consalign_spss_turner + consalign_spss_trained, "Structural alignment scoring model": ["Ensemble"] * len(consalign_spss_ensemble) + ["Turner"] * len(consalign_spss_turner) + ["Trained"] * len(consalign_spss_trained)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "Structural alignment scoring model", y = "Sum-of-pairs score", data = data_frame, sym = "")
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/consalign_model_comparison_sps_scoring_model_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"F1 score": consalign_f1_scores_trained + consalign_f1_scores_trained_random_init + consalign_f1_scores_transferred_only, "Scoring parameter training type": ["Transfer-learned"] * len(consalign_f1_scores_trained) + ["Random-learned"] * len (consalign_f1_scores_trained_random_init) + ["Transferred only"] * len(consalign_f1_scores_transferred_only)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "Scoring parameter training type", y = "F1 score", data = data_frame)
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/consalign_model_comparison_f1_score_train_type_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Matthews correlation coefficient": consalign_mccs_trained + consalign_mccs_trained_random_init + consalign_mccs_transferred_only, "Scoring parameter training type": ["Transfer-learned"] * len(consalign_mccs_trained) + ["Random-learned"] * len(consalign_mccs_trained_random_init) + ["Transferred only"] * len(consalign_mccs_transferred_only)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "Scoring parameter training type", y = "Matthews correlation coefficient", data = data_frame)
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/consalign_model_comparison_mcc_train_type_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Sum-of-pairs score": consalign_spss_trained + consalign_spss_trained_random_init + consalign_spss_transferred_only, "Scoring parameter training type": ["Transfer-learned"] * len(consalign_spss_trained) + ["Random-learned"] * len(consalign_spss_trained_random_init) + ["Transferred only"] * len(consalign_spss_transferred_only)}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.boxplot(x = "Scoring parameter training type", y = "Sum-of-pairs score", data = data_frame, sym = "")
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/consalign_model_comparison_sps_train_type_box_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Average F1 score": [consalign_f1_score_ensemble, raf_f1_score, locarna_f1_score, dafs_f1_score, sparse_f1_score, turbofold_f1_score], "RNA structural aligner": ["ConsAlign", "RAF", "LocARNA ", "DAFS", "SPARSE", "Linear\nTurboFold"]}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "RNA structural aligner", y = "Average F1 score", data = data_frame)
  fig = ax.get_figure()
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/rna_aligner_f1_scores.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Average MCC": [consalign_mcc_ensemble, raf_mcc, locarna_mcc, dafs_mcc, sparse_mcc, turbofold_mcc], "RNA structural aligner": ["ConsAlign", "RAF", "LocARNA ", "DAFS", "SPARSE", "Linear\nTurboFold"]}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "RNA structural aligner", y = "Average MCC", data = data_frame)
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/rna_aligner_mccs.eps", bbox_inches = "tight")
  pyplot.clf()
  data = {"Average sum-of-pairs score": [consalign_sps_ensemble, raf_sps, locarna_sps, dafs_sps, sparse_sps, turbofold_sps], "RNA structural aligner": ["ConsAlign", "RAF", "LocARNA ", "DAFS", "SPARSE", "Linear\nTurboFold"]}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "RNA structural aligner", y = "Average sum-of-pairs score", data = data_frame)
  fig = ax.get_figure()
  fig.tight_layout()
  fig.savefig(image_dir_path + "/rna_aligner_spss.eps", bbox_inches = "tight")
  fig.clf()
  f1_scores = [consalign_f1_score_ensemble, consalign_f1_score_turner, consalign_f1_score_trained]
  data = {"Average F1 score": f1_scores, "Structural alignment scoring model": ["Ensemble", "Turner", "Trained"],}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "Structural alignment scoring model", y = "Average F1 score", data = data_frame)
  fig = ax.get_figure()
  fig.tight_layout()
  fig.savefig(image_dir_path + "/consalign_model_comparison_f1_score_scoring_model.eps", bbox_inches = "tight")
  fig.clf()
  mccs = [consalign_mcc_ensemble, consalign_mcc_turner, consalign_mcc_trained]
  data = {"Average MCC": mccs, "Structural alignment scoring model": ["Ensemble", "Turner", "Trained"],}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "Structural alignment scoring model", y = "Average MCC", data = data_frame)
  fig = ax.get_figure()
  fig.tight_layout()
  fig.savefig(image_dir_path + "/consalign_model_comparison_mcc_scoring_model.eps", bbox_inches = "tight")
  fig.clf()
  spss = [consalign_sps_ensemble, consalign_sps_turner, consalign_sps_trained]
  data = {"Average sum-of-pairs score": spss, "Structural alignment scoring model": ["Ensemble", "Turner", "Trained"],}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "Structural alignment scoring model", y = "Average sum-of-pairs score", data = data_frame)
  fig = ax.get_figure()
  fig.tight_layout()
  fig.savefig(image_dir_path + "/consalign_model_comparison_sps_scoring_model.eps", bbox_inches = "tight")
  fig.clf()
  f1_scores = [consalign_f1_score_trained, consalign_f1_score_trained_random_init, consalign_f1_score_transferred_only]
  data = {"Average F1 score": f1_scores, "Scoring parameter training type": ["Transfer-learned", "Random-learned", "Transferred only"],}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "Scoring parameter training type", y = "Average F1 score", data = data_frame)
  fig = ax.get_figure()
  fig.tight_layout()
  fig.savefig(image_dir_path + "/consalign_model_comparison_f1_score_train_type.eps", bbox_inches = "tight")
  fig.clf()
  mccs = [consalign_mcc_trained, consalign_mcc_trained_random_init, consalign_mcc_transferred_only]
  data = {"Average MCC": mccs, "Scoring parameter training type": ["Transfer-learned", "Random-learned", "Transferred only"],}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "Scoring parameter training type", y = "Average MCC", data = data_frame)
  fig = ax.get_figure()
  fig.tight_layout()
  fig.savefig(image_dir_path + "/consalign_model_comparison_mcc_train_type.eps", bbox_inches = "tight")
  fig.clf()
  spss = [consalign_sps_trained, consalign_sps_trained_random_init, consalign_sps_transferred_only]
  data = {"Average sum-of-pairs score": spss, "Scoring parameter training type": ["Transfer-learned", "Random-learned", "Transferred only"],}
  data_frame = pandas.DataFrame(data = data)
  ax = seaborn.barplot(x = "Scoring parameter training type", y = "Average sum-of-pairs score", data = data_frame)
  fig = ax.get_figure()
  fig.tight_layout()
  fig.savefig(image_dir_path + "/consalign_model_comparison_sps_train_type.eps", bbox_inches = "tight")
  fig.clf()

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
  if total > 0.:
    return tp / total
  else:
    return float('-inf')

def get_f1_score(result):
  (tp, tn, fp, fn) = result
  denom = tp + 0.5 * (fp + fn)
  if denom > 0.:
    return tp / denom
  else:
    return float('-inf')

def get_mcc(result):
  (tp, tn, fp, fn) = result
  denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if denom > 0.:
    return (tp * tn - fp * fn) / denom
  else:
    return float('-inf')

if __name__ == "__main__":
  main()
