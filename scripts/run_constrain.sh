#!/bin/zsh
#$ -S /bin/zsh
#$ -l epyc -l s_vmem=64G -l mem_req=64G
#$ -N run_constrain
#$ -V
#$ -b y
#$ -w e
#$ -o constrain_output_log.dat
#$ -e constrain_error_log.dat
#$ -m ea
#$ -M heartsh@heartsh.io
#$ -cwd

time constrain -i ../assets/train_data -o ../assets/costs.dat
