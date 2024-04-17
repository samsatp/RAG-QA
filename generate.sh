#!/bin/bash

#SBATCH -J generating
#SBATCH -o csc_log_generating/%j.out
#SBATCH -e csc_log_generating/%j.err
#SBATCH -A project_2001403
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH -n 6
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sathianpong.trangcasanchai@helsinki.fi


echo "Starting at `date`"


module purge
source venv/bin/activate

python main.py -c generating \
               -question_with_context_file $1 \
               -use_context $2 \
               -use_gold $3


echo "Finishing at `date`"