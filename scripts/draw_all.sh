#! /usr/bin/env sh

./draw_train_logs.py \
  && ./get_stats_of_ss_estimation_programs.py \
  && ./get_stats_of_ss_estimation_programs_rnastralign.py \
  && ./get_stats_of_ss_estimation_programs_bralibase.py \
  && ./get_stats_of_ss_estimation_programs_2.py \
  && ./get_stats_of_ss_estimation_programs_rnastralign_2.py \
  && ./get_stats_of_ss_estimation_programs_bralibase_2.py \
  && ./get_stats_of_ss_estimation_programs_3.py \
  && ./get_stats_of_ss_estimation_programs_rnastralign_3.py \
  && ./get_stats_of_ss_estimation_programs_bralibase_3.py
