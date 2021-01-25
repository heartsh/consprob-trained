#!/bin/sh
#$ -S /bin/sh
#$ -l medium -l s_vmem=64G -l mem_req=64G
#$ -pe smp 128
#$ -N run_rtrain
#$ -V
#$ -b y
#$ -w e
#$ -o rtrain_output_log.dat
#$ -e rtrain_error_log.dat
#$ -m ea
#$ -M heartsh@heartsh.io
#$ -cwd

rtrain -i assets/train_data -o assets/costs.dat
