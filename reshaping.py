#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:37:57 2022

@author: Janus Lammert ☺
"""

import numpy as np
import scipy as sp
import math

if __name__ == "__main__":
    print("execute main.py")

def Window(data, windowsize=15, num_win=50, dis_win=0, WindowFunction='flattop'):
    """
    This function creates a windows with a flattop shape. The windows are sliced along the x axis and span the hole y axis.
    Theey get calculated for all Images included in the data Array.

    Parameters
    ----------
    data : 3D Array
        Images with the first index indexing the image.
    windowsize : Interger, optional
        This is the Size of the window. Note that the usable windowsize is at 
        around one third of this value. It is a symetric function within the 
        window. The default is 15.
    num_win : Integer, optional
        Number of windows. Note that only num_win OR dis_win can be set 
        otherwise you will receive an ERROR. It is recommended to use num_win 
        over dis_win. The default is 32.
    dis_win : float32, optional
        Distance of two windows. (From startingpoint to startingpoint) Note 
        that only num_win OR dis_win can be set otherwise you will receive an 
        ERROR. The default is 0.
    WindowFunction : String, optional
        flattop: Für Fourier optimiert
        flat: schneidet einfach nur Daten aus

    Returns
    -------
    result: 4D Array
        The first index indexes the image the window was taken from. The second index indexes the windows.

    """
    if dis_win == 0:
        lenz, leny, lenx = np.shape(data)
        dis_win = (lenx - windowsize)/num_win
    elif num_win != 32 and dis_win != 0:
        print('ERROR: Pleas only set num_win or dis_win')
    else:
        num_win = math.floor((lenx - windowsize)/dis_win)
    if WindowFunction == 'flattop':
        window_func = sp.signal.windows.flattop(windowsize)
    elif WindowFunction == 'flat':
        window_func = np.ones(windowsize)
    else:
        print('Please use a viable windowfunction')
    result = np.zeros((lenz, num_win, leny, windowsize))
#    printProgressBar(0, num_win*lenz, prefix='Progress:',
#                     suffix='Complete', length=50)
    for wn in range(num_win):
        for i in range(lenz):
            for y in range(leny):
                result[i, wn, y, :] = data[i, y, int(
                    wn*dis_win):int(wn*dis_win+windowsize)]*window_func
#            printProgressBar(wn*lenz+i+1, num_win*lenz,
#                             prefix='Progress:', suffix='Complete', length=50)
    return result


def ReduceSize(data, red_x=0, red_y=0):
    """
    Reduces the size of images by a given amount. can accept 2D, 3D and $D arrays with the last two indices indexing the pixels.

    Parameters
    ----------
    data : 2D, 3D, 4D Array
        Input image (set).
    red_x : Integer, optional
        By how much should the imagesize be reduced on both sides in x direction. The default is 0.
    red_y : Integer, optional
        By how much should the imagesize be reduced on both sides in y direction. The default is 0.

    Returns
    -------
    reduced_img : 2D, 3D, 4D Array
        The reduced images.

    """
    dim = len(np.shape(data))
    if red_x != 0 and red_y != 0:
        if dim == 2:
            reduced_img = data[red_y:-1*red_y, red_x:-1*red_x]
        elif dim == 3:
            reduced_img = data[:, red_y:-1*red_y, red_x:-1*red_x]
        elif dim == 4:
            reduced_img = data[:, :, red_y:-1*red_y, red_x:-1*red_x]
        else:
            print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
    elif red_x == 0 and red_y != 0:
        if dim == 2:
            reduced_img = data[red_y:-1*red_y, :]
        elif dim == 3:
            reduced_img = data[:, red_y:-1*red_y, :]
        elif dim == 4:
            reduced_img = data[:, :, red_y:-1*red_y, :]
        else:
            print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
    elif red_x != 0 and red_y == 0:
        if dim == 2:
            reduced_img = data[:, red_x:-1*red_x]
        elif dim == 3:
            reduced_img = data[:, :, red_x:-1*red_x]
        elif dim == 4:
            reduced_img = data[:, :, :, red_x:-1*red_x]
        else:
            print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
    return reduced_img

def reduce_matrix(matrix, indices):
    # Convert indices array to set for faster lookup
    indices_set = set(indices)
    
    # Create a mask for selecting rows and columns based on indices
    mask = np.array([i in indices_set for i in range(len(matrix))])
    
    # Select rows and columns based on the mask
    reduced_matrix = matrix[mask][:, mask]
    
    return reduced_matrix

def delete_rows(input_array, indices):
    """
    Select rows from a 2D NumPy array based on the provided indices.

    Parameters:
    - input_array (numpy.ndarray): Input 2D NumPy array.
    - indices (list or numpy.ndarray): Array of row indices to select.

    Returns:
    - numpy.ndarray: New array containing only the selected rows.
    """
    # Convert indices to NumPy array if it's not already
    indices = np.array(indices)

    # Select rows based on indices
    selected_rows = input_array[indices]

    return selected_rows
