#! /usr/bin/env python

import compile_rna_fams
import compile_rna_fams_rnastralign
import compile_rna_fams_bralibase
import infernal_check_rnastralign
import infernal_check_bralibase
import draw_train_logs
import run_ss_estimation_programs
import run_ss_estimation_programs_rnastralign
import run_ss_estimation_programs_bralibase
import get_stats_of_ss_estimation_programs
import get_stats_of_ss_estimation_programs_rnastralign
import get_stats_of_ss_estimation_programs_bralibase
import get_stats_of_ss_estimation_programs_2
import get_stats_of_ss_estimation_programs_rnastralign_2
import get_stats_of_ss_estimation_programs_bralibase_2
import get_stats_of_ss_estimation_programs_3
import get_stats_of_ss_estimation_programs_rnastralign_3
import get_stats_of_ss_estimation_programs_bralibase_3

def main():
  compile_rna_fams.main()
  compile_rna_fams_rnastralign.main()
  compile_rna_fams_bralibase.main()
  infernal_check_rnastralign.main()
  infernal_check_bralibase.main()
  draw_train_logs.main()
  run_ss_estimation_programs.main()
  run_ss_estimation_programs_rnastralign.main()
  run_ss_estimation_programs_bralibase.main()
  get_stats_of_ss_estimation_programs.main()
  get_stats_of_ss_estimation_programs_rnastralign.main()
  get_stats_of_ss_estimation_programs_bralibase.main()
  get_stats_of_ss_estimation_programs_2.main()
  get_stats_of_ss_estimation_programs_rnastralign_2.main()
  get_stats_of_ss_estimation_programs_bralibase_2.main()
  get_stats_of_ss_estimation_programs_3.main()
  get_stats_of_ss_estimation_programs_rnastralign_3.main()
  get_stats_of_ss_estimation_programs_bralibase_3.main()

if __name__ == "__main__":
  main()
