#!/bin/sh
#$ -S /bin/sh
#$ -l epyc -l s_vmem=32G -l mem_req=32G
#$ -N run_constrain
#$ -V
#$ -b y
#$ -w e
#$ -o constrain_output_log.dat
#$ -e constrain_error_log.dat
#$ -m ea
#$ -M heartsh@heartsh.io
#$ -cwd

constrain -i assets/train_data -o assets/costs.dat
