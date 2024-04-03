#!/bin/bash

# Check if the user has provided a filename
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "File $1 not found!"
    exit 1
fi

# Sort the file based on the value in the sixth column
sort -k6nr "$1" > sorted_"$1"

echo "File sorted based on the sixth column. Sorted file is saved as sorted_$1"


