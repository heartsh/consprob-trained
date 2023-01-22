import os
import string
import subprocess
from Bio import AlignIO
from statistics import mode, mean
import numpy
from Bio import SeqIO
import time

bracket_pairs = [("(", ")"), ("<", ">"), ("{", "}"), ("[", "]"), ("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"), ("E", "e"), ]

def get_dir_paths():
  current_work_dir_path = os.getcwd()
  (head, tail) = os.path.split(current_work_dir_path)
  asset_dir_path = head + "/assets"
  program_dir_path = "/usr/local" if current_work_dir_path.find("/home/masaki") == -1 else "/home/masaki/prgrms"
  conda_program_dir_path = "/usr/local/ancnd/envs/rsrch" if current_work_dir_path.find("/home/masaki") == -1 else "/home/masaki/prgrms/ancnd/envs/rsrch"
  return (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path)

def run_raf(raf_params):
  (rna_file_path, raf_output_file_path) = raf_params
  seq_length_avg, seq_num = get_seq_info(rna_file_path)
  raf_command = "raf predict " + rna_file_path + " --weights $(cat ../assets/raf_trained_weights.dat)"
  begin = time.time()
  (output, _, _) = run_command(raf_command)
  elapsed_time = time.time() - begin
  raf_output_file = open(raf_output_file_path, "w+")
  raf_output_file.write(output.decode())
  raf_output_file.close()
  sta = AlignIO.read(raf_output_file_path, "fasta")
  recs = sta[:-1]
  new_sta = AlignIO.MultipleSeqAlignment(recs)
  new_sta.column_annotations["secondary_structure"] = str(sta[-1].seq)
  AlignIO.write(new_sta, raf_output_file_path, "stockholm")
  return (seq_length_avg, seq_num, elapsed_time)

def run_dafs(dafs_params):
  (rna_file_path, dafs_output_file_path) = dafs_params
  seq_length_avg, seq_num = get_seq_info(rna_file_path)
  dafs_command = "dafs " + rna_file_path
  begin = time.time()
  (output, _, _) = run_command(dafs_command)
  elapsed_time = time.time() - begin
  dafs_output_file = open(dafs_output_file_path, "w+")
  dafs_output_file.write(output.decode())
  dafs_output_file.close()
  sta = AlignIO.read(dafs_output_file_path, "fasta")
  recs = sta[1:]
  new_sta = AlignIO.MultipleSeqAlignment(recs)
  new_sta.column_annotations["secondary_structure"] = str(sta[0].seq)
  AlignIO.write(new_sta, dafs_output_file_path, "stockholm")
  return (seq_length_avg, seq_num, elapsed_time)

def run_locarna(locarna_params):
  (rna_file_path, locarna_output_file_path, is_sparse) = locarna_params
  seq_length_avg, seq_num = get_seq_info(rna_file_path)
  locarna_command = "mlocarna " + rna_file_path + " --keep-sequence-order --width=10000"
  if is_sparse:
    locarna_command += " --sparse"
  begin = time.time()
  (output, _, _) = run_command(locarna_command)
  elapsed_time = time.time() - begin
  lines = [line.strip() for (i, line) in enumerate(str(output).split("\\n")) if i > 7]
  locarna_output_file = open(locarna_output_file_path, "w+")
  locarna_output_buf = "# STOCKHOLM 1.0\n\n"
  for line in lines:
    if line.startswith("alifold "):
      locarna_output_buf += "#=GC SS_cons %s\n//" % line.split()[1]
      break
    else:
      locarna_output_buf += line + "\n"
  locarna_output_file.write(locarna_output_buf)
  locarna_output_file.close()
  return (seq_length_avg, seq_num, elapsed_time)

def run_consalign(consalign_params):
  (rna_file_path, consalign_output_dir_path, sub_thread_num, scoring_model, train_type, disables_alifold, disables_transplant) = consalign_params
  seq_length_avg, seq_num = get_seq_info(rna_file_path)
  consalign_command = "consalign %s%s-t " % ("-d " if disables_alifold else "", "-p " if disables_transplant else "") + str(sub_thread_num) + " -i " + rna_file_path + " -o " + consalign_output_dir_path + " -m " + scoring_model + " -u " + train_type
  begin = time.time()
  run_command(consalign_command)
  elapsed_time = time.time() - begin
  return (seq_length_avg, seq_num, elapsed_time)

def run_turbofold(turbofold_params):
  (rna_file_path, turbofold_output_dir_path) = turbofold_params
  seq_length_avg, seq_num = get_seq_info(rna_file_path)
  turbofold_command = "linearturbofold -i %s -o %s" % (rna_file_path, turbofold_output_dir_path)
  begin = time.time()
  run_command(turbofold_command)
  elapsed_time = time.time() - begin
  recs = [rna_seq.seq for rna_seq in SeqIO.parse(rna_file_path, "fasta")]
  num_of_recs = len(recs)
  lines = [""] * (2 * num_of_recs)
  for i in range(num_of_recs):
    lines[2 * i] = ">%d\n" % i
  for ss_file in os.listdir(turbofold_output_dir_path):
    if not ss_file.endswith(".db"):
      continue
    ss_file_path = os.path.join(turbofold_output_dir_path, ss_file)
    reader = open(ss_file_path, "r")
    ss = reader.readlines()[-1].strip()
    reader.close()
    (ss_name, extension) = os.path.splitext(ss_file)
    i = int(ss_name.split("_")[0]) - 1
    lines[2 * i + 1] = ss + "\n\n"
  ss_output_file_path = turbofold_output_dir_path + "/output.fa"
  ss_output_file = open(ss_output_file_path, "w")
  ss_output_file.writelines(lines)
  ss_output_file.close()
  return (seq_length_avg, seq_num, elapsed_time)

def get_seq_info(rna_file_path):
  rnas = SeqIO.parse(rna_file_path, "fasta")
  rna_lens = [len(rec.seq) for rec in rnas]
  rna_len_avg = mean(rna_lens)
  num_of_rnas = len(rna_lens)
  return rna_len_avg, num_of_rnas

def write_elapsed_time_data(elapsed_time_data, output_file_path):
  data_strings = []
  for item in elapsed_time_data:
    seq_len_avg, seq_num, elapsed_time = item
    string = "%f,%d,%f\n" % (seq_len_avg, seq_num, elapsed_time)
    data_strings.append(string)
  with open(output_file_path, "w") as f:
    f.writelines(data_strings)

def read_elapsed_time_data(input_file_path):
  seq_lens_avg, seq_nums, elapsed_times, = [], [], [],
  with open(input_file_path, "r") as f:
    for line in f.readlines():
      seq_len_avg, seq_num, elapsed_time = line.split(",")
      seq_lens_avg.append(float(seq_len_avg))
      seq_nums.append(int(seq_num))
      elapsed_times.append(float(elapsed_time))
  return seq_lens_avg, seq_nums, elapsed_times,

def get_css(css_file_path):
  sta = AlignIO.read(css_file_path, "stockholm")
  css_string = sta.column_annotations["secondary_structure"]
  sta_len = len(sta[0])
  num_of_rnas = len(sta)
  pos_map_sets = []
  for i in range(num_of_rnas):
    pos_map_sets.append([])
    pos = -1
    for j in range(sta_len):
      char = sta[i][j]
      if char != "-":
        pos += 1
      if char != "-":
        pos_map_sets[i].append(pos)
      else:
        pos_map_sets[i].append(-1)
  css = []
  for i in range(num_of_rnas):
    css.append({})
  stack = []
  for (left, right) in bracket_pairs:
    for i, char in enumerate(css_string):
      if char == left:
        stack.append(i)
      elif char == right:
        col_pos = stack.pop()
        for j in range(num_of_rnas):
          base_pair_1 = (sta[j][col_pos], sta[j][i])
          if base_pair_1[0] == "-" or base_pair_1[1] == "-":
            continue
          pos_pair_1 = (pos_map_sets[j][col_pos], pos_map_sets[j][i])
          css[j][pos_pair_1] = True
  return css

def run_command(command):
  subproc = subprocess.Popen(
    command,
    stdout = subprocess.PIPE,
    shell = True
  )
  (output, error) = subproc.communicate()
  returned_code = subproc.wait()
  return (output, error, returned_code)

def get_sss(ss_file_path):
  sss = []
  ss_strings = []
  ss_strings = [rec.seq for rec in SeqIO.parse(ss_file_path, "fasta")]
  sss = []
  for (i, ss_string) in enumerate(ss_strings):
    sss.append({})
    for (left, right) in bracket_pairs:
      stack = []
      for (j, char) in enumerate(ss_string):
        if char == left:
          stack.append(j)
        elif char == right:
          pos = stack.pop()
          sss[i][(pos, j)] = True
  return sss

def get_sss_ct(ss_dir_path, seq_num):
  sss = []
  for i in range(seq_num):
    ss_file = "%d.ct" % i
    ss_file_path = os.path.join(ss_dir_path, ss_file)
    ss = read_ct_file(ss_file_path)
    sss.append(ss)
  return sss

def read_ct_file(ss_file_path):
  ss = {}
  with open(ss_file_path) as ss_file:
    for line in ss_file.readlines():
      line = line.strip()
      splits = line.split()
      if len(splits) != 6:
        continue
      (left_partner, right_partner) = (int(splits[0]), int(splits[4]))
      if right_partner == 0:
        continue
      ss[(left_partner - 1, right_partner - 1)] = True
  return ss
