#!/bin/bash

for question_with_context_file in RAG/results/retrieval_file_*.xlsx
do
  echo "$question_with_context_file"
  sbatch generate.sh "$question_with_context_file" y
done