#! /usr/bin/env python

import utils
from Bio import SeqIO
from Bio import AlignIO
import numpy
import seaborn
from matplotlib import pyplot
import os
import multiprocessing
import time
import datetime
import shutil
from os import path

gammas = [2. ** i for i in range(-7, 11)]

def main():
  (current_work_dir_path, asset_dir_path, program_dir_path, conda_program_dir_path) = utils.get_dir_paths()
  num_of_threads = multiprocessing.cpu_count()
  temp_dir_path = "/tmp/run_ss_estimation_programs_%s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
  if not os.path.isdir(temp_dir_path):
    os.mkdir(temp_dir_path)
  mafft_xinsi_params = []
  conshomfold_params = []
  conshomfold_params_4_running_time = []
  conshomfold_old_params = []
  conshomfold_old_params_4_running_time = []
  centroidhomfold_params = []
  centroidhomfold_params_4_running_time = []
  rnafold_params = []
  contrafold_params = []
  contrafold_params_4_running_time = []
  centroidfold_params = []
  centroidfold_params_4_running_time = []
  contextfold_params = []
  rnaalifold_params = []
  rnaalifold_params_4_running_time = []
  centroidalifold_params = []
  centroidalifold_params_4_running_time = []
  petfold_params = []
  petfold_params_4_running_time = []
  mafft_xinsi_dir_path = asset_dir_path + "/mafft_xinsi"
  conshomfold_dir_path = asset_dir_path + "/conshomfold"
  conshomfold_old_dir_path = asset_dir_path + "/conshomfold_old"
  centroidhomfold_dir_path = asset_dir_path + "/centroidhomfold"
  rnafold_dir_path = asset_dir_path + "/rnafold"
  contrafold_dir_path = asset_dir_path + "/contrafold"
  centroidfold_dir_path = asset_dir_path + "/centroidfold"
  contextfold_dir_path = asset_dir_path + "/contextfold"
  rnaalifold_dir_path = asset_dir_path + "/rnaalifold"
  centroidalifold_dir_path = asset_dir_path + "/centroidalifold"
  petfold_dir_path = asset_dir_path + "/petfold"
  infernal_black_list_dir_path = asset_dir_path + "/infernal_black_list"
  if not os.path.isdir(mafft_xinsi_dir_path):
    os.mkdir(mafft_xinsi_dir_path)
  if not os.path.isdir(conshomfold_dir_path):
    os.mkdir(conshomfold_dir_path)
  if not os.path.isdir(conshomfold_old_dir_path):
    os.mkdir(conshomfold_old_dir_path)
  if not os.path.isdir(centroidhomfold_dir_path):
    os.mkdir(centroidhomfold_dir_path)
  if not os.path.isdir(rnafold_dir_path):
    os.mkdir(rnafold_dir_path)
  if not os.path.isdir(contrafold_dir_path):
    os.mkdir(contrafold_dir_path)
  if not os.path.isdir(centroidfold_dir_path):
    os.mkdir(centroidfold_dir_path)
  if not os.path.isdir(contextfold_dir_path):
    os.mkdir(contextfold_dir_path)
  if not os.path.isdir(rnaalifold_dir_path):
    os.mkdir(rnaalifold_dir_path)
  if not os.path.isdir(centroidalifold_dir_path):
    os.mkdir(centroidalifold_dir_path)
  if not os.path.isdir(petfold_dir_path):
    os.mkdir(petfold_dir_path)
  rna_dir_path = asset_dir_path + "/test_data"
  sub_thread_num = 4
  for rna_file in os.listdir(rna_dir_path):
    if not rna_file.endswith(".fa"):
      continue
    rna_file_path = os.path.join(rna_dir_path, rna_file)
    (rna_family_name, extension) = os.path.splitext(rna_file)
    mafft_xinsi_output_file_path = os.path.join(mafft_xinsi_dir_path, rna_family_name + ".aln")
    mafft_xinsi_output_file_path_2 = os.path.join(mafft_xinsi_dir_path, rna_family_name + ".fa")
    conshomfold_output_dir_path = os.path.join(conshomfold_dir_path, rna_family_name)
    conshomfold_old_output_dir_path = os.path.join(conshomfold_old_dir_path, rna_family_name)
    centroidhomfold_output_dir_path = os.path.join(centroidhomfold_dir_path, rna_family_name)
    rnafold_output_file_path = os.path.join(rnafold_dir_path, rna_family_name + ".fa")
    contrafold_output_dir_path = os.path.join(contrafold_dir_path, rna_family_name)
    centroidfold_output_dir_path = os.path.join(centroidfold_dir_path, rna_family_name)
    contextfold_output_file_path = os.path.join(contextfold_dir_path, rna_family_name + ".fa")
    rnaalifold_output_file_path = os.path.join(rnaalifold_dir_path, rna_family_name + ".fa")
    centroidalifold_output_dir_path = os.path.join(centroidalifold_dir_path, rna_family_name)
    petfold_output_dir_path = os.path.join(petfold_dir_path, rna_family_name)
    infernal_black_list_file_path = os.path.join(infernal_black_list_dir_path, rna_family_name + "_infernal.dat")
    if os.path.isfile(infernal_black_list_file_path):
      continue
    if not os.path.isdir(conshomfold_output_dir_path):
      os.mkdir(conshomfold_output_dir_path)
    if not os.path.isdir(conshomfold_old_output_dir_path):
      os.mkdir(conshomfold_old_output_dir_path)
    if not os.path.isdir(centroidhomfold_output_dir_path):
      os.mkdir(centroidhomfold_output_dir_path)
    if not os.path.isdir(contrafold_output_dir_path):
      os.mkdir(contrafold_output_dir_path)
    if not os.path.isdir(centroidfold_output_dir_path):
      os.mkdir(centroidfold_output_dir_path)
    if not os.path.isdir(centroidalifold_output_dir_path):
      os.mkdir(centroidalifold_output_dir_path)
    if not os.path.isdir(petfold_output_dir_path):
      os.mkdir(petfold_output_dir_path)
    mafft_xinsi_params.insert(0, (rna_file_path, mafft_xinsi_output_file_path))
    conshomfold_params.insert(0, (sub_thread_num, rna_file_path, conshomfold_output_dir_path, True, False))
    conshomfold_params_4_running_time.insert(0, (sub_thread_num, rna_file_path, conshomfold_output_dir_path, True, True))
    conshomfold_old_params.insert(0, (sub_thread_num, rna_file_path, conshomfold_old_output_dir_path, False, False))
    conshomfold_old_params_4_running_time.insert(0, (sub_thread_num, rna_file_path, conshomfold_old_output_dir_path, False, True))
    rnafold_params.insert(0, (rna_file_path, rnafold_output_file_path))
    contextfold_params.insert(0, (rna_file_path, contextfold_output_file_path, temp_dir_path))
    rnaalifold_params.insert(0, (mafft_xinsi_output_file_path, rnaalifold_output_file_path))
    for gamma in gammas:
      gamma_str = str(gamma) if gamma < 1 else str(int(gamma))
      output_file = "gamma=" + gamma_str + ".fa"
      petfold_gamma_str = str(1 / gamma) if gamma > 1 else str(int(1 / gamma))
      centroidhomfold_output_file_path = os.path.join(centroidhomfold_output_dir_path, output_file)
      centroidhomfold_params.insert(0, (rna_file_path, centroidhomfold_output_file_path, gamma_str, temp_dir_path))
      if gamma == 1.:
        centroidhomfold_params_4_running_time.insert(0, (rna_file_path, centroidhomfold_output_file_path, gamma_str, temp_dir_path))
      contrafold_output_file_path = os.path.join(contrafold_output_dir_path, output_file)
      contrafold_params.insert(0, (rna_file_path, contrafold_output_file_path, gamma_str, temp_dir_path))
      if gamma == 1.:
        contrafold_params_4_running_time.insert(0, (rna_file_path, contrafold_output_file_path, gamma_str, temp_dir_path))
      centroidfold_output_file_path = os.path.join(centroidfold_output_dir_path, output_file)
      centroidfold_params.insert(0, (rna_file_path, centroidfold_output_file_path, gamma_str))
      if gamma == 1.:
        centroidfold_params_4_running_time.insert(0, (rna_file_path, centroidfold_output_file_path, gamma_str))
      centroidalifold_output_file_path = os.path.join(centroidalifold_output_dir_path, output_file)
      centroidalifold_params.insert(0, (mafft_xinsi_output_file_path, centroidalifold_output_file_path, gamma_str))
      if gamma == 1.:
        centroidalifold_params_4_running_time.insert(0, (mafft_xinsi_output_file_path, centroidalifold_output_file_path, gamma_str))
      petfold_output_file_path = os.path.join(petfold_output_dir_path, output_file)
      petfold_params.insert(0, (mafft_xinsi_output_file_path_2, petfold_output_file_path, petfold_gamma_str))
      if gamma == 1.:
        petfold_params_4_running_time.insert(0, (mafft_xinsi_output_file_path_2, petfold_output_file_path, petfold_gamma_str))
  pool = multiprocessing.Pool(num_of_threads)
  # pool.map(run_mafft_xinsi, mafft_xinsi_params)
  pool = multiprocessing.Pool(int(num_of_threads / sub_thread_num))
  pool.map(run_conshomfold, conshomfold_params)
  if False:
    begin = time.time()
    # pool.map(utils.run_command, conshomfold_params_4_running_time)
    pool.map(run_conshomfold, conshomfold_params_4_running_time)
    conshomfold_elapsed_time = time.time() - begin
  pool.map(run_conshomfold, conshomfold_old_params)
  if False:
    begin = time.time()
    # pool.map(utils.run_command, conshomfold_params_4_running_time)
    pool.map(run_conshomfold, conshomfold_old_params_4_running_time)
    conshomfold_old_elapsed_time = time.time() - begin
  pool = multiprocessing.Pool(num_of_threads)
  pool.map(run_centroidhomfold, centroidhomfold_params)
  if False:
    begin = time.time()
    pool.map(run_centroidhomfold, centroidhomfold_params_4_running_time)
    centroidhomfold_elapsed_time = time.time() - begin
  if False:
    begin = time.time()
    pool.map(run_rnafold, rnafold_params)
    rnafold_elapsed_time = time.time() - begin
    pool.map(run_contrafold, contrafold_params)
  if False:
    begin = time.time()
    pool.map(run_contrafold, contrafold_params_4_running_time)
    contrafold_elapsed_time = time.time() - begin
  # pool.map(run_centroidfold, centroidfold_params)
  if False:
    begin = time.time()
    pool.map(run_centroidfold, centroidfold_params_4_running_time)
    centroidfold_elapsed_time = time.time() - begin
  if False:
    begin = time.time()
    pool.map(run_contextfold, contextfold_params)
    contextfold_elapsed_time = time.time() - begin
  if False:
    begin = time.time()
    pool.map(run_rnaalifold, rnaalifold_params)
    rnaalifold_elapsed_time = time.time() - begin
    pool.map(run_centroidalifold, centroidalifold_params)
  if False:
    begin = time.time()
    pool.map(run_centroidalifold, centroidalifold_params_4_running_time)
    centroidalifold_elapsed_time = time.time() - begin
  # pool.map(run_petfold, petfold_params)
  if False:
    begin = time.time()
    pool.map(run_petfold, petfold_params_4_running_time)
    petfold_elapsed_time = time.time() - begin
  if False:
    print("The elapsed time of ConsHomfold (new) = %f [s]." % conshomfold_elapsed_time)
    print("The elapsed time of ConsHomfold (old) = %f [s]." % conshomfold_old_elapsed_time)
    print("The elapsed time of CentroidHomfold = %f [s]." % centroidhomfold_elapsed_time)
    print("The elapsed time of RNAfold = %f [s]." % rnafold_elapsed_time)
    print("The elapsed time of CONTRAfold = %f [s]." % contrafold_elapsed_time)
    print("The elapsed time of CentroidFold = %f [s]." % centroidfold_elapsed_time)
    print("The elapsed time of Contextfold = %f [s]." % contextfold_elapsed_time)
    print("The elapsed time of RNAalifold = %f [s]." % rnaalifold_elapsed_time)
    print("The elapsed time of CentroidAlifold = %f [s]." % centroidalifold_elapsed_time)
    print("The elapsed time of PETfold = %f [s]." % petfold_elapsed_time)
  shutil.rmtree(temp_dir_path)

def run_rnafold(rnafold_params):
  (rna_file_path, rnafold_output_file_path) = rnafold_params
  rnafold_command = "RNAfold --noPS -i " + rna_file_path
  (output, _, _) = utils.run_command(rnafold_command)
  lines = [str(line) for line in str(output).strip().split("\\n") if line.startswith(".") or line.startswith("(")]
  buf = ""
  for i, line in enumerate(lines):
    buf += ">%d\n%s\n" % (i, line.split()[0])
  rnafold_output_file = open(rnafold_output_file_path, "w")
  rnafold_output_file.write(buf)
  rnafold_output_file.close()

def run_centroidfold(centroidfold_params):
  (rna_file_path, centroidfold_output_file_path, gamma_str) = centroidfold_params
  centroidfold_command = "centroid_fold " + rna_file_path + " -g " + gamma_str
  (output, _, _) = utils.run_command(centroidfold_command)
  lines = [line.split()[0] for (i, line) in enumerate(str(output).split("\\n")) if i % 3 == 2]
  centroidfold_output_file = open(centroidfold_output_file_path, "w+")
  centroidfold_output_buf = ""
  for (i, line) in enumerate(lines):
    centroidfold_output_buf += ">%d\n%s\n\n" % (i, line)
  centroidfold_output_file.write(centroidfold_output_buf)
  centroidfold_output_file.close()

def run_contrafold(contrafold_params):
  (rna_file_path, contrafold_output_file_path, gamma_str, temp_dir_path) = contrafold_params
  contrafold_output_file = open(contrafold_output_file_path, "w+")
  contrafold_output_buf = ""
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  seq_file_path = os.path.join(temp_dir_path, "seq_4_%s_and_gamma=%s.fa" % (rna_family_name, gamma_str))
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  for (i, rec) in enumerate(recs):
    SeqIO.write([rec], open(seq_file_path, "w"), "fasta")
    contrafold_command = "contrafold predict " + seq_file_path + " --gamma " + gamma_str
    (output, _, _) = utils.run_command(contrafold_command)
    contrafold_output_buf += ">%d\n%s\n\n" % (i, str(output).strip().split("\\n")[3])
  contrafold_output_file.write(contrafold_output_buf)
  contrafold_output_file.close()

def run_centroidhomfold(centroidhomfold_params):
  (rna_file_path, centroidhomfold_output_file_path, gamma_str, temp_dir_path) = centroidhomfold_params
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  rec_seq_len = len(recs)
  centroidhomfold_output_file = open(centroidhomfold_output_file_path, "w+")
  centroidhomfold_output_buf = ""
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  seq_file_path = os.path.join(temp_dir_path, "seqs_4_%s_and_gamma=%s.fa" % (rna_family_name, gamma_str))
  hom_seq_file_path = os.path.join(temp_dir_path, "hom_seqs_4_%s_and_gamma=%s.fa" % (rna_family_name, gamma_str))
  for (i, rec) in enumerate(recs):
    SeqIO.write([rec], open(seq_file_path, "w"), "fasta")
    hom_recs = [rec for (j, rec) in enumerate(recs) if j != i]
    SeqIO.write(recs, open(hom_seq_file_path, "w"), "fasta")
    centroidhomfold_command = "centroid_homfold " + seq_file_path + " -H " + hom_seq_file_path + " -g " + gamma_str
    (output, _, _) = utils.run_command(centroidhomfold_command)
    centroidhomfold_output_buf += ">%d\n%s\n\n" % (i, str(output).split("\\n")[2].split()[0])
  centroidhomfold_output_file.write(centroidhomfold_output_buf)
  centroidhomfold_output_file.close()

def run_contextfold(contextfold_params):
  (rna_file_path, contextfold_output_file_path, temp_dir_path) = contextfold_params
  recs = [rec for rec in SeqIO.parse(rna_file_path, "fasta")]
  basename = os.path.basename(rna_file_path)
  (rna_family_name, extension) = os.path.splitext(basename)
  seq_file_path = os.path.join(temp_dir_path, "seqs_4_%s.txt" % rna_family_name)
  tmp_output_buf = ""
  for rec in recs:
    tmp_output_buf += str(rec.seq) + "\n"
  seq_file = open(seq_file_path, "w+")
  seq_file.write(tmp_output_buf)
  seq_file.close()
  contextfold_command = "java -cp bin contextFold.app.Predict in:" + seq_file_path
  (_, _, _) = utils.run_command(contextfold_command)
  with open(seq_file_path + ".pred") as f:
    sss = [line.strip() for (i, line) in enumerate(f.readlines()) if (i + 1) % 3 == 2]
  contextfold_output_buf = ""
  for (i, ss) in enumerate(sss):
    contextfold_output_buf += ">%d\n%s\n\n" % (i, ss)
  contextfold_output_file = open(contextfold_output_file_path, "w+")
  contextfold_output_file.write(contextfold_output_buf)
  contextfold_output_file.close()

def run_conshomfold(conshomfold_params):
  (sub_thread_num, rna_file_path, conshomfold_output_dir_path, is_trained, is_benchmarked) = conshomfold_params
  conshomfold_command = "conshomfold%s %s-t " % ("-trained" if is_trained else "", "-b " if is_benchmarked else "") + str(sub_thread_num) + " -i " + rna_file_path + " -o " + conshomfold_output_dir_path
  utils.run_command(conshomfold_command)

def run_rnaalifold(rnaalifold_params):
  (sa_file_path, rnaalifold_output_file_path) = rnaalifold_params
  rnaalifold_command = "RNAalifold -q --noPS " + sa_file_path
  (output, _, _) = utils.run_command(rnaalifold_command)
  css = str(output).strip().split("\\n")[1].split()[0]
  sta = AlignIO.read(sa_file_path, "clustal")
  AlignIO.write(sta, rnaalifold_output_file_path, "stockholm")
  sta = AlignIO.read(rnaalifold_output_file_path, "stockholm")
  sta.column_annotations["secondary_structure"] = css
  sss = get_sss(sta)
  buf = ""
  for (i, ss) in enumerate(sss):
    buf += ">%d\n%s\n\n" % (i, ss)
  rnaalifold_output_file = open(rnaalifold_output_file_path, "w+")
  rnaalifold_output_file.write(buf)
  rnaalifold_output_file.close()

def run_centroidalifold(centroidalifold_params):
  (sa_file_path, centroidalifold_output_file_path, gamma_str) = centroidalifold_params
  centroidalifold_command = "centroid_alifold " + sa_file_path + " -g " + gamma_str
  (output, _, _) = utils.run_command(centroidalifold_command)
  css = str(output).strip().split("\\n")[2].split()[0]
  sta = AlignIO.read(sa_file_path, "clustal")
  AlignIO.write(sta, centroidalifold_output_file_path, "stockholm")
  sta = AlignIO.read(centroidalifold_output_file_path, "stockholm")
  sta.column_annotations["secondary_structure"] = css
  sss = get_sss(sta)
  buf = ""
  for (i, ss) in enumerate(sss):
    buf += ">%d\n%s\n\n" % (i, ss)
  centroidalifold_output_file = open(centroidalifold_output_file_path, "w+")
  centroidalifold_output_file.write(buf)
  centroidalifold_output_file.close()

def run_petfold(petfold_params):
  (sa_file_path, petfold_output_file_path, gamma_str) = petfold_params
  petfold_command = "PETfold -f " + sa_file_path + " -a " + gamma_str
  (output, _, _) = utils.run_command(petfold_command)
  css = str(output).strip().split("\\n")[2].split("\\t")[1].strip()
  sta = AlignIO.read(sa_file_path, "fasta")
  AlignIO.write(sta, petfold_output_file_path, "stockholm")
  sta = AlignIO.read(petfold_output_file_path, "stockholm")
  sta.column_annotations["secondary_structure"] = css
  sss = get_sss(sta)
  buf = ""
  for (i, ss) in enumerate(sss):
    buf += ">%d\n%s\n\n" % (i, ss)
  petfold_output_file = open(petfold_output_file_path, "w+")
  petfold_output_file.write(buf)
  petfold_output_file.close()

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

def run_mafft_xinsi(mafft_xinsi_params):
  (rna_seq_file_path, mafft_xinsi_output_file_path) = mafft_xinsi_params
  mafft_xinsi_command = "mafft-xinsi --thread 1 --quiet " + "--clustalout " + rna_seq_file_path + " > " + mafft_xinsi_output_file_path
  utils.run_command(mafft_xinsi_command)
  sa = AlignIO.read(mafft_xinsi_output_file_path, "clustal")
  mafft_xinsi_output_file_path = os.path.splitext(mafft_xinsi_output_file_path)[0] + ".fa"
  AlignIO.write(sa, mafft_xinsi_output_file_path, "fasta")

if __name__ == "__main__":
  main()