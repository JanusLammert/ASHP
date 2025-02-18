#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:12:21 2022

@author: Janus Lammert ☺
"""

import pandas as pd
import starfile
import numpy as np

def dataframes_to_dict(df1, df2):
    """
    Takes two pandas DataFrames and returns a dictionary with the specified keys.

    Parameters:
    df1 (pandas.DataFrame): First DataFrame
    df2 (pandas.DataFrame): Second DataFrame

    Returns:
    dict: A dictionary with keys 'OPTICS' and 'data' mapping to df1 and df2, respectively.
    """
    data_dict = {'optics': df1, 'particles': df2}
    return data_dict

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

def split_dataframe_by_classes(df, labels, tt):
    n_dataframes = int(np.max(labels)) + 1
    dataframes = [[] for _ in range(n_dataframes)]
    for index, row in df.iterrows():
        class_n = int(row['rlnClassNumber']) - 1
        dataframe_assignment = labels[np.where(tt == class_n)][0]
        dataframes[dataframe_assignment].append(row)
    # Convert lists of rows to DataFrames
    dataframes = [pd.DataFrame(rows, columns=df.columns) for rows in dataframes]
    return dataframes

def split_dataframe_by_fibrils(df, labels):
    n_dataframes = int(np.max(labels)) + 1
    dataframes = [[] for _ in range(n_dataframes)]
    for index, row in df.iterrows():
        fibril_n = int(row['rlnNewIndices']) - 1
        dataframe_assignment = labels[fibril_n]
        dataframes[dataframe_assignment].append(row)
    # Convert lists of rows to DataFrames
    dataframes = [pd.DataFrame(rows, columns=df.columns) for rows in dataframes]
    return dataframes

def read_class_averages_star(filename):
    try:
        # Read the star file using starfile library
        star_data = starfile.read(filename)

        # Extract _rlnClassNumber column
        class_numbers = star_data['rlnClassNumber'].astype(int)
        
        translation_table = np.asarray(class_numbers.tolist()) - 1

        return translation_table
    except Exception as e:
        print("An error occurred:", e)
        return []
