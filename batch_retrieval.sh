#!/bin/bash


# Execute the file list generator and capture its output in a temporary file
tmp_file=$(mktemp)
python utils.py ls_collections > "$tmp_file"

# Iterate through the filenames in the temporary file
while IFS= read -r collection; do
    sbatch retrieval.sh  "data/covid/not_bad_dev.xlsx" 10 "$collection"
done < "$tmp_file"

# Remove the temporary file
rm "$tmp_file"
