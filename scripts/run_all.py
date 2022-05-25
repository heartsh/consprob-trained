#! /usr/bin/env python

import compile_rna_fams
import compile_rna_fams_rnastralign
import compile_rna_fams_bralibase
import draw_epochs_vs_costs
import run_ss_estimation_programs
import run_ss_estimation_programs_rnastralign
import run_ss_estimation_programs_bralibase
import get_stats_of_ss_estimation_programs
import get_stats_of_ss_estimation_programs_rnastralign
import get_stats_of_ss_estimation_programs_bralibase
import get_stats_of_ss_estimation_programs_2
import get_stats_of_ss_estimation_programs_rnastralign_2
import get_stats_of_ss_estimation_programs_bralibase_2

def main():
  compile_rna_fams.main()
  draw_epochs_vs_costs.main()
  run_ss_estimation_programs.main()
  get_stats_of_ss_estimation_programs.main()

if __name__ == "__main__":
  main()
