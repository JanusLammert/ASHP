#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:12:21 2022

@author: Janus Lammert ☺
"""

import pandas as pd
import starfile
import numpy as np

def create_empty_dataframes(df, n):
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n should be a positive integer.")

    # Get column names from the input DataFrame
    columns = df.columns.tolist()

    # Create a list of empty DataFrames
    empty_dataframes = [pd.DataFrame(columns=columns) for _ in range(n)]

    return empty_dataframes

def split_dataframe_by_array(df, assignment_array):
    # Check if the length of the dataframe and the assignment array are the same
    if len(df) != len(assignment_array):
        raise ValueError("Length of the DataFrame and assignment array must be the same.")

    # Get unique values in the assignment array to determine the number of dataframes
    k = len(set(assignment_array))

    # Create a dictionary to store dataframes based on assignments
    dataframes_dict = {i: pd.DataFrame(columns=df.columns) for i in range(1, k + 1)}

    # Iterate through the DataFrame and assign each entry to the corresponding dataframe
    for index, assignment in enumerate(assignment_array):
        dataframes_dict[assignment] = dataframes_dict[assignment].append(df.iloc[index])

    # Convert the dictionary of dataframes to a list of dataframes
    result_dataframes = list(dataframes_dict.values())

    return result_dataframes

def write_dataframes_to_starfile(dataframe1, dataframe2, output_filename):
    # Convert DataFrames to dictionaries for starfile writing
    dict1 = dataframe1.to_dict(orient='list')
    dict2 = dataframe2.to_dict(orient='list')

    # Create a starfile writer
    with starfile.open(output_filename, 'w') as sf:
        # Write the first DataFrame
        sf.write({f"{col}_1": values for col, values in dict1.items()})

        # Write the second DataFrame
        sf.write({f"{col}_2": values for col, values in dict2.items()})


def split_dataframe_by_classes(df, labels):
    n_dataframes = np.max(labels)
    dataframes = create_empty_dataframes(df, n_dataframes)
    for index, row in df.iterrows():
        class_n = row['rlnClassNumber']
        dataframe_assignment = labels[class_n]
        dataframes[dataframe_assignment] = dataframes[dataframe_assignment].append(row, ignore_index=True)
    return dataframes

def split_dataframe_by_fibrils(df, labels):
    n_dataframes = np.max(labels)
    dataframes = create_empty_dataframes(df, n_dataframes)
    for index, row in df.iterrows():
        fibril_n = row['rlnNewIndices']
        dataframe_assignment = labels[fibril_n]
        dataframes[dataframe_assignment] = dataframes[dataframe_assignment].append(row, ignore_index=True)
    return dataframes

