#!/bin/bash

#SBATCH -J retrieval
#SBATCH -o csc_log_retrieval/%j.out
#SBATCH -e csc_log_retrieval/%j.err
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

python main.py -c retrieval \
               -question_file $1 \
               -k $2 \
               -collection_name $3


echo "Finishing at `date`"