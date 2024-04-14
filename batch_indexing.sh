#!/bin/bash


embedding_model="BAAI/bge-base-en-v1.5 "
distance_fn="cosine"
chunk_sizes=(300 500 700 1000)
chunk_overlaps=(0.0 0.3 0.5)

for chunk_size in "${chunk_sizes[@]}"; do
    for chunk_overlap in "${chunk_overlaps[@]}"; do
        sbatch indexing.sh $embedding_model $distance_fn $chunk_size $chunk_overlap
    done
done