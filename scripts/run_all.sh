#! /usr/bin/env sh

./compile_rna_fams.py \
  && ./compile_rna_fams_rnastralign.py \
  && ./compile_rna_fams_bralibase.py \
  && ./infernal_check_rnastralign.py \
  && ./infernal_check_bralibase.py \
  && ./draw_train_logs.py \
  && ./run_ss_estimation_programs.py \
  && ./run_ss_estimation_programs_rnastralign.py \
  && ./run_ss_estimation_programs_bralibase.py \
  && ./get_stats_of_ss_estimation_programs.py \
  && ./get_stats_of_ss_estimation_programs_rnastralign.py \
  && ./get_stats_of_ss_estimation_programs_bralibase.py \
  && ./get_stats_of_ss_estimation_programs_2.py \
  && ./get_stats_of_ss_estimation_programs_rnastralign_2.py \
  && ./get_stats_of_ss_estimation_programs_bralibase_2.py \
  && ./get_stats_of_ss_estimation_programs_3.py \
  && ./get_stats_of_ss_estimation_programs_rnastralign_3.py \
  && ./get_stats_of_ss_estimation_programs_bralibase_3.py
