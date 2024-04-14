#!/bin/bash

#SBATCH -J indexing
#SBATCH -o csc_log_indexing/%j.out
#SBATCH -e csc_log_indexing/%j.err
#SBATCH -A project_2001403
#SBATCH -t 5:00:00
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sathianpong.trangcasanchai@helsinki.fi


echo "Starting at `date`"


module purge
source venv/bin/activate

embedding_model="thenlper/gte-base"
distance_fn="cosine"
chunk_sizes=(300 500 700 1000)
chunk_overlaps=(0.0 0.3 0.5)

for chunk_size in "${chunk_sizes[@]}"; do
    for chunk_overlap in "${chunk_overlaps[@]}"; do
        sbatch indexing.sh $embedding_model $distance_fn $chunk_size $chunk_overlap
    done
done


echo "Finishing at `date`"