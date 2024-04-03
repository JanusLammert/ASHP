#!/bin/bash

# Loop through files starting with "sbat" in the current directory
for file in sbat*; do
    # Check if the file exists and is a regular file
    if [ -f "$file" ]; then
        echo "Executing sbatch for file: $file"
        # Execute sbatch followed by the filename
        sbatch "$file"
    fi
done

