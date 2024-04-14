#!/bin/bash

#SBATCH -J indexing
#SBATCH -o csc_log_indexing/%j.out
#SBATCH -e csc_log_indexing/%j.err
#SBATCH -A project_2001403
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -n 6
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sathianpong.trangcasanchai@helsinki.fi


echo "Starting at `date`"


module purge
source venv/bin/activate

python main.py -c indexing \
               -embedding_model $1 \
               -distance_fn $2 \
               -chunk_size $3 \
               -chunk_overlap $4


echo "Finishing at `date`"