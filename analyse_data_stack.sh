#!/bin/bash

# Check if the number of arguments provided is correct
if [ $# -ne 2 ]; then
    echo "Usage: $0 <folder> <output_file>"
    exit 1
fi

# Directory containing files
folder=$1

# Check if the directory exists
if [ ! -d "$folder" ]; then
    echo "Directory '$folder' does not exist."
    exit 1
fi

output_file=$2

# Function to extract column values from filename
extract_columns() {
    filename="$1"
    if [[ $filename == *MSE* ]]; then
        col1="MSD"
    elif [[ $filename == *conv* ]]; then
        col1="convolution"
    fi
    col2=""
    if [[ $filename == *_l_* ]]; then
        col2="laplacian"
    else
	col2="plane"
    fi
    col3=$(echo "$filename" | grep -oP '(?<=_w)\d+')
    col4=$(echo "$filename" | grep -oP '(?<=_n)\d+')
    col5=$(echo "$filename" | grep -oP '(?<=_t)\d+')
}

# Create the output file and add headers
echo -e "Col1\tCol2\tCol3\tCol4\tCol5\tOutput" > "$output_file"

# Iterate over files in the directory
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        # Extract column values from filename
        extract_columns "$file"

        # Check if t is less than or equal to n
        if [ "$col5" -le "$col4" ]; then
            # Execute python script and store output
            output=$(python3 analyse_2.py -n 5 -m $col1 -t $col5 -i "$file")

            # Output the table
            echo -e "$col1\t$col2\t$col3\t$col4\t$col5\t$output" >> "$output_file"
        fi
    fi
done
