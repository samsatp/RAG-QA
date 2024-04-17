#!/bin/bash

for answer_file in RAG/results/answer_file_*.xlsx
do
  echo "$answer_file"
  sbatch evaluate_learned.sh "$answer_file" y
done

