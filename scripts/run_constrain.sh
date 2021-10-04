#!/bin/zsh
#$ -S /bin/zsh
#$ -l epyc -l s_vmem=128G -l mem_req=128G
#$ -N run_constrain
#$ -V
#$ -b y
#$ -w e
#$ -o constrain_output_log.dat
#$ -e constrain_error_log.dat
#$ -m ea
#$ -M heartsh@heartsh.io
#$ -cwd

time constrain -i ../assets/train_data_sa -j ../assets/train_data_ss -o ../assets/costs_sa.dat  -p ../assets/costs_ss.dat
