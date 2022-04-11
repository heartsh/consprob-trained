import os
import string
import subprocess
from Bio import AlignIO
from statistics import mode
import numpy
from Bio import SeqIO

bracket_pairs = [("(", ")"), ("<", ">"), ("{", "}"), ("[", "]"), ("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"), ("E", "e"), ]

def get_dir_paths():
  current_work_dir_path = os.getcwd()
  (head, tail) = os.path.split(current_work_dir_path)
  asset_dir_path = head + "/assets"
  program_dir_path = "/usr/local" if current_work_dir_path.find("/home/masaki") == -1 else "/home/masaki/prgrms"
  conda_program_dir_path = "/usr/local/ancnd/envs/rsrch" if current_work_dir_path.find("/home/masaki") == -1 else "/home/masaki/prgrms/ancnd/envs/rsrch"
  return (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path)

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
  if ss_file_path.endswith(".fa"):
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
  else:
    sss = read_bpseq_file(ss_file_path)
  return sss

def read_bpseq_file(ss_file_path):
  sss = []
  idx = 0
  with open(ss_file_path) as ss_file:
    for line in ss_file.readlines():
      line = line.strip()
      if len(line) == 0 or line.startswith("#") or "accuracy=" in line:
        continue
      if line.startswith("1 "):
        sss.append({})
        idx += 1
      splits = line.split()
      (left_partner, right_partner) = (int(splits[0]), int(splits[2]))
      if right_partner == 0:
        continue
      sss[idx - 1][(left_partner - 1, right_partner - 1)] = True
  return sss
