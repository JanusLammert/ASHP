#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:26:23 2023

@author: janus
"""

# Import the necessary modules
import starfile
import pandas as pd
import numpy as np

def normalize_rows(matrix):
        """
        Normalize every row in the matrix.

        Parameters:
        - matrix: 2D NumPy array

        Returns:
        - normalized_matrix: 2D NumPy array with normalized rows
        """
        row_sums = matrix.sum(axis=1)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        return normalized_matrix

def normalize_columns(matrix):
    """
    Normalize every column in the matrix.

    Parameters:
    - matrix: 2D NumPy array

    Returns:
    - normalized_matrix: 2D NumPy array with normalized columns
    """
    col_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / col_sums
    return normalized_matrix


def chep(path='/home/janus/Jureca/230814_red_tau/Select/job068/particles.star'):

    # Read the Star file and create two DataFrames for optics and particle data
    star = starfile.read(path)
    optics = pd.DataFrame.from_dict(star["optics"])
    particles = pd.DataFrame.from_dict(star["particles"])

    # Create an empty list for micrograph names
    Micrographs = []

    # Iterate through particle rows and add unique micrograph names to the list
    for index, row in particles.iterrows():
        Micrograph = row['rlnMicrographName']
        
        if Micrograph not in Micrographs:
            Micrographs.append(Micrograph)

    # Create a list of IDs for micrographs
    ID = list(range(len(Micrographs)))

    # Create an empty DataFrame with micrograph names as columns
    df = pd.DataFrame(columns=Micrographs)

    # Fill the first row of the DataFrame with empty lists for each micrograph
    df.loc[1] = [[] for _ in range(len(Micrographs))]
    # Fill the second row of the DataFrame with zeros for each micrograph
    df.loc[2] = [0 for _ in range(len(Micrographs))]
    # Fill the third row of the DataFrame with zeros for each micrograph
    df.loc[3] = [0 for _ in range(len(Micrographs))]

    # Iterate through particle rows and add helix IDs to the corresponding lists in the first row of the DataFrame
    for index, row in particles.iterrows():
        Micrograph = row['rlnMicrographName']
        HelixID = row['rlnHelicalTubeID']
        
        if HelixID not in df.loc[1, Micrograph]:
            df.loc[1, Micrograph].append(HelixID)

    # Calculate the number of helices for each micrograph and save it in the second row of the DataFrame
    for Micrograph in Micrographs:
        df.loc[2, Micrograph] = len(df.loc[1, Micrograph])
        
    # Calculate the cumulative number of helices for each micrograph and save it in the third row of the DataFrame
    temp = 0
    for Micrograph in Micrographs:
        df.loc[3, Micrograph] = df.loc[2, Micrograph] + temp
        temp += df.loc[2, Micrograph]

    # Get the number of rows and columns of the particle DataFrame
    rows, col = particles.shape
    # Create an empty array for the new indices of the helices
    new_indices = np.zeros(rows)

    # Iterate through particle rows and calculate the new indices of the helices based on the data in the DataFrame
    for index, row in particles.iterrows():
        Micrograph = row['rlnMicrographName']
        HelixID = row['rlnHelicalTubeID']
        
        new_indices[index] = 1 + df.loc[3, Micrograph] - df.loc[2, Micrograph] + df.loc[1, Micrograph].index(HelixID)
            
    # Get the maximum number of helices and the maximum number of classes from the particle DataFrame
    number_helices = np.max(new_indices)
    number_classes = particles['rlnClassNumber'].max()

    # Add the new indices of the helices as a new column to the particle DataFrame
    particles = particles.assign(rlnNewIndices=new_indices)

    # Create an empty array for the matrix of classes and helices
    Matrix = np.zeros((int(number_classes), int(number_helices)))  
            
    # Iterate through particle rows and increment the corresponding elements of the matrix by one
    for index, row in particles.iterrows():
        ClassID = row['rlnClassNumber']
        HelixID = row['rlnNewIndices']
        
        Matrix[int(ClassID-1), int(HelixID-1)] += 1

    matrix_class_norm = normalize_rows(Matrix)
    matrix_helix_norm = normalize_columns(Matrix)

    return matrix_class_norm, matrix_helix_norm