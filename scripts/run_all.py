#! /usr/bin/env python

import compile_rna_fams
import draw_epochs_vs_costs
import run_ss_estimation_programs
import get_stats_of_ss_estimation_programs

def main():
  compile_rna_fams.main()
  draw_epochs_vs_costs.main()
  run_ss_estimation_programs.main()
  get_stats_of_ss_estimation_programs.main()

if __name__ == "__main__":
  main()
