#! /usr/bin/env python

import utils
from Bio import SeqIO
import numpy
import seaborn
from matplotlib import pyplot
import os
import multiprocessing
import time
import datetime
import shutil
from os import path
from mpl_toolkits.mplot3d import Axes3D
from Bio import AlignIO
import random

running_time_ratio_thres = 90
color_palette = seaborn.color_palette()
color = color_palette[3]

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  temp_dir_path = "/tmp/run_ss_estimation_programs_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  conshomfold_params = []
  contrafold_params = []
  raf_params = []
  conshomfold_dir_path = asset_dir_path + "/conshomfold"
  contrafold_dir_path = asset_dir_path + "/contrafold"
  raf_dir_path = asset_dir_path + "/raf"
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  if not os.path.isdir(conshomfold_dir_path):
    os.mkdir(conshomfold_dir_path)
  if not os.path.isdir(contrafold_dir_path):
    os.mkdir(contrafold_dir_path)
  if not os.path.isdir(raf_dir_path):
    os.mkdir(raf_dir_path)
  rna_dir_path = asset_dir_path + "/all_data"
  sub_thread_num = 4
  for rna_file in os.listdir(rna_dir_path):
    if not rna_file.endswith(".fa"):
      continue
    rna_file_path = os.path.join(rna_dir_path, rna_file)
    (rna_family_name, extension) = os.path.splitext(rna_file)
    conshomfold_output_dir_path = os.path.join(conshomfold_dir_path, rna_family_name)
    contrafold_output_dir_path = os.path.join(contrafold_dir_path, rna_family_name)
    if not os.path.isdir(contrafold_output_dir_path):
      os.mkdir(contrafold_output_dir_path)
    contrafold_output_file_path = os.path.join(contrafold_dir_path, rna_family_name + ".fa")
    raf_output_file_path = os.path.join(raf_dir_path, rna_family_name + ".fa")
    infernal_black_list_file_path = os.path.join(infernal_black_list_dir_path, rna_family_name + "_infernal.dat")
    if os.path.isfile(infernal_black_list_file_path):
      continue
    if not os.path.isdir(conshomfold_output_dir_path):
      os.mkdir(conshomfold_output_dir_path)
    lens = [len(str(rec.seq)) for rec in SeqIO.parse(rna_file_path, "fasta")]
    max_seq_len = max(lens)
    seq_num = len(lens)
    conshomfold_command = "conshomfold -b -t 1 -i " + rna_file_path + " -o " + conshomfold_output_dir_path
    conshomfold_params.insert(0, (max_seq_len, seq_num, conshomfold_command))
    contrafold_params.insert(0, (max_seq_len, seq_num, rna_file_path, contrafold_output_file_path, "1", temp_dir_path))
    raf_params.insert(0, (max_seq_len, seq_num, rna_file_path, raf_output_file_path))
  pool = multiprocessing.Pool(num_of_threads)
  results = pool.map(run_conshomfold, conshomfold_params)
  output_file_path = asset_dir_path + "/conshomfold_running_time_3d_plot.dat"
  buf = ""
  for result in results:
    buf += "%d,%d,%f\n" % (result[0], result[1], result[2])
  with open(output_file_path, "w") as f:
    f.write(buf)
  max_seq_lens = []
  seq_nums = []
  running_times = []
  max_seq_lens_4_conshomfold = []
  seq_nums_4_conshomfold = []
  running_times_4_conshomfold = []
  with open(output_file_path) as f:
    for line in f.readlines():
      split = line.strip().split(",")
      (max_seq_len, seq_num, running_time) = (int(split[0]), int(split[1]), float(split[2]))
      if seq_num >= running_time_ratio_thres:
        max_seq_lens_4_conshomfold.append(max_seq_len)
        seq_nums_4_conshomfold.append(seq_num)
        running_times_4_conshomfold.append(running_time)
      else:
        max_seq_lens.append(max_seq_len)
        seq_nums.append(seq_num)
        running_times.append(running_time)
  # y = numpy.random.rand(len(max_seq_lens)) * 2 - 1
  y = numpy.random.rand(len(max_seq_lens), 3) 
  fig = pyplot.figure()
  ax = fig.add_subplot(111, projection="3d")
  plot = ax.scatter(max_seq_lens, seq_nums, running_times, zdir = 'z', s = 50, c = y, cmap = pyplot.cm.jet)
  plot_2 = ax.scatter(max_seq_lens_4_conshomfold, seq_nums_4_conshomfold, running_times_4_conshomfold, zdir = 'z', s = 50, facecolor = color, edgecolor = "black", linestyle = "solid", label = "# RNA homologs $\geq$ %d" % running_time_ratio_thres, marker = "s")
  ax.set_xlabel("Maximum sequence length")
  ax.set_ylabel("# RNA homologs")
  ax.set_zlabel("Running time (sec)")
  image_dir_path = asset_dir_path + "/images"
  if not os.path.exists(image_dir_path):
    os.mkdir(image_dir_path)
  pyplot.legend(loc = "upper left")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/conshomfold_running_time_3d_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  results = pool.map(run_contrafold, contrafold_params)
  output_file_path = asset_dir_path + "/contrafold_running_time_3d_plot.dat"
  buf = ""
  for result in results:
    buf += "%d,%d,%f\n" % (result[0], result[1], result[2])
  with open(output_file_path, "w") as f:
    f.write(buf)
  max_seq_lens = []
  seq_nums = []
  running_times = []
  max_seq_lens_4_contrafold = []
  seq_nums_4_contrafold = []
  running_times_4_contrafold = []
  with open(output_file_path) as f:
    for line in f.readlines():
      split = line.strip().split(",")
      (max_seq_len, seq_num, running_time) = (int(split[0]), int(split[1]), float(split[2]))
      if seq_num >= running_time_ratio_thres:
        max_seq_lens_4_contrafold.append(max_seq_len)
        seq_nums_4_contrafold.append(seq_num)
        running_times_4_contrafold.append(running_time)
      else:
        max_seq_lens.append(max_seq_len)
        seq_nums.append(seq_num)
        running_times.append(running_time)
  fig = pyplot.figure()
  ax = fig.add_subplot(111, projection="3d")
  plot = ax.scatter(max_seq_lens, seq_nums, running_times, zdir = 'z', s = 50, c = y, cmap = pyplot.cm.jet)
  itr_len = len(running_times_4_contrafold)
  speedup_ratio = round(sum(map(get_ratio, zip(running_times_4_conshomfold, running_times_4_contrafold))) / itr_len)
  plot_2 = ax.scatter(max_seq_lens_4_contrafold, seq_nums_4_contrafold, running_times_4_contrafold, zdir = 'z', s = 50, facecolor = color, edgecolor = "black", linestyle = "solid", label = "On average, CONTRAfold is %d-fold faster than ConsHomfold" % speedup_ratio, marker = "s")
  ax.set_xlabel("Maximum sequence length")
  ax.set_ylabel("# RNA homologs")
  ax.set_zlabel("Running time (sec)")
  pyplot.legend(loc = "upper left")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/contrafold_running_time_3d_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  results = pool.map(run_raf, raf_params)
  output_file_path = asset_dir_path + "/raf_running_time_3d_plot.dat"
  buf = ""
  for result in results:
    buf += "%d,%d,%f\n" % (result[0], result[1], result[2])
  with open(output_file_path, "w") as f:
    f.write(buf)
  max_seq_lens = []
  seq_nums = []
  running_times = []
  max_seq_lens_4_raf = []
  seq_nums_4_raf = []
  running_times_4_raf = []
  with open(output_file_path) as f:
    for line in f.readlines():
      split = line.strip().split(",")
      (max_seq_len, seq_num, running_time) = (int(split[0]), int(split[1]), float(split[2]))
      if seq_num >= running_time_ratio_thres:
        max_seq_lens_4_raf.append(max_seq_len)
        seq_nums_4_raf.append(seq_num)
        running_times_4_raf.append(running_time)
      else:
        max_seq_lens.append(max_seq_len)
        seq_nums.append(seq_num)
        running_times.append(running_time)
  fig = pyplot.figure()
  ax = fig.add_subplot(111, projection="3d")
  plot = ax.scatter(max_seq_lens, seq_nums, running_times, zdir = 'z', s = 50, c = y, cmap = pyplot.cm.jet)
  speedup_ratio = round(sum(map(get_ratio, zip(running_times_4_raf, running_times_4_conshomfold))) / itr_len)
  plot_2 = ax.scatter(max_seq_lens_4_raf, seq_nums_4_raf, running_times_4_raf, zdir = 'z', s = 50, facecolor = color, edgecolor = "black", linestyle = "solid", label = "On average, ConsHomfold is %d-fold faster than RAF" % speedup_ratio, marker = "s")
  ax.set_xlabel("Maximum sequence length")
  ax.set_ylabel("# RNA homologs")
  ax.set_zlabel("Running time (sec)")
  pyplot.legend(loc = "upper left")
  pyplot.tight_layout()
  pyplot.savefig(image_dir_path + "/raf_running_time_3d_plot.eps", bbox_inches = "tight")
  pyplot.clf()
  shutil.rmtree(temp_dir_path)

def get_ratio(pair):
  return pair[0] / pair[1]

def run_conshomfold(conshomfold_params):
  (max_seq_len, seq_num, params) = conshomfold_params
  begin = time.time()
  utils.run_command(params)
  elapsed_time = time.time() - begin
  return (max_seq_len, seq_num, elapsed_time)

def run_contrafold(contrafold_params):
  (max_seq_len, seq_num, rna_file_path, contrafold_output_file_path, gamma_str, temp_dir_path) = contrafold_params
  contrafold_output_file = open(contrafold_output_file_path, "w+")
  contrafold_output_buf = ""
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  seq_file_path = os.path.join(temp_dir_path, "seq_4_%s_and_gamma=%s.fa" % (rna_family_name, gamma_str))
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  begin = time.time()
  for (i, rec) in enumerate(recs):
    SeqIO.write([rec], open(seq_file_path, "w"), "fasta")
    contrafold_command = "contrafold predict " + seq_file_path + " --gamma " + gamma_str
    (output, _, _) = utils.run_command(contrafold_command)
    contrafold_output_buf += ">%d\n%s\n\n" % (i, str(output).strip().split("\\n")[3])
  elapsed_time = time.time() - begin
  contrafold_output_file.write(contrafold_output_buf)
  contrafold_output_file.close()
  return (max_seq_len, seq_num, elapsed_time)

def run_raf(raf_params):
  (max_seq_len, seq_num, rna_file_path, raf_output_file_path) = raf_params
  raf_command = "raf predict " + rna_file_path
  begin = time.time()
  (output, _, _) = utils.run_command(raf_command)
  elapsed_time = time.time() - begin
  raf_output_file = open(raf_output_file_path, "w+")
  raf_output_file.write(output.decode())
  raf_output_file.close()
  sta = AlignIO.read(raf_output_file_path, "fasta")
  recs = sta[:-1]
  new_sta = AlignIO.MultipleSeqAlignment(recs)
  new_sta.column_annotations["secondary_structure"] = str(sta[-1].seq)
  sss = get_sss(new_sta)
  buf = ""
  raf_output_file = open(raf_output_file_path, "w+")
  for (i, ss) in enumerate(sss):
    buf += ">%d\n%s\n\n" % (i, ss)
  raf_output_file.write(buf)
  raf_output_file.close()
  return (max_seq_len, seq_num, elapsed_time)

def get_sss(sta):
  css_string = sta.column_annotations["secondary_structure"]
  sta_len = len(sta[0])
  num_of_rnas = len(sta)
  pos_map_sets = []
  sss = []
  for i in range(num_of_rnas):
    pos_map_sets.append([])
    pos = -1
    for j in range(sta_len):
      char = sta[i][j]
      if char != "-":
        pos += 1
      pos_map_sets[i].append(pos)
    sss.append("." * (pos + 1))
  stack = []
  for i, char in enumerate(css_string):
    if char == "(" or char == "<" or char == "[" or char == "{":
      stack.append(i)
    elif char == ")" or char == ">" or char == "]" or char == "}":
      col_pos = stack.pop()
      for j in range(num_of_rnas):
        base_pair_1 = (sta[j][col_pos], sta[j][i])
        if base_pair_1[0] == "-" or base_pair_1[1] == "-":
          continue
        pos_pair_1 = (pos_map_sets[j][col_pos], pos_map_sets[j][i])
        sss[j] = sss[j][:pos_pair_1[0]] + "(" + sss[j][pos_pair_1[0] + 1:]
        sss[j] = sss[j][:pos_pair_1[1]] + ")" + sss[j][pos_pair_1[1] + 1:]
  return sss

if __name__ == "__main__":
  main()
