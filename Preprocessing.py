#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:31:31 2022

@author: Janus Lammert ☺
"""

import numpy as np
import scipy as sp

if __name__ == "__main__":
    print("execute main.py")

def laplacian(data):
    """
    Calculates the laplacian of the images.

    Parameters
    ----------
    data : 3D Array
        Stack of the images the laplacian gets calculated of.

    Returns
    -------
    result : 3D Array
        Stack of the laplacians.

    """
    lenz, leny, lenx = np.shape(data)
    result = np.copy(data)
    for i in range(lenz):
        stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        result[i, :, :] = sp.ndimage.convolve(
            data[i, :, :], stencil, mode='nearest')
    return result

def y_projections(data):
    """
    Calculates the projection onto the y-axis. If one wants to apply this function onto multiple images, one can input them as a stacked image.

    Parameters
    ----------
    data : Array (2D or 3D)
        An Array with the data of which the projection gets calculated. In case of a 3 dimensional array, the first index is the index of the image.

    Returns
    -------
    projections : Array (1D or 2D)
        The resulting Projection as an array.

    """
    dim = len(np.shape(data))
    if dim == 3:
        lenz, leny, lenx = np.shape(data)
        projections = np.zeros((lenz, leny))
        for i in range(0, lenz):
            for y in range(0, leny):
                projections[i, y] = np.sum(data[i, y, :])
    elif dim == 2:
        leny, lenx = np.shape(data)
        projections = np.zeros(leny)
        for y in range(0, leny):
            projections[y] = np.sum(data[y, :])
    else:
        print('ERROR: Input a 2 or 3 dimensional Array into the y_projections() function')

    return projections

def meanpixelbrightness(data):
    """
    Calculates the mean brightness if the images. It receives a 3D array with the first index indexing the images.

    Parameters
    ----------
    data : 3D Array
        Stacked images as input data.

    Returns
    -------
    brightness : 1D Array
        Mean pixelvalue for each image.

    """
    lenz, leny, lenx = np.shape(data)
    brightness = np.zeros(lenz)
    for i in range(lenz):
        brightness[i] = np.sum(data[i, :, :])
    brightness /= lenx*leny
    return brightness


def MSE(data):
    """
    Calculates the MSE of the images with one and another.

    Parameters
    ----------
    data : 3D Array
        First index idixes the image.

    Returns
    -------
    err : 2D Array
        Results of the MSE culculation with the indices bing the indices of the image.

    """
    lenz, leny, lenx = np.shape(data)
    err = np.zeros((lenz, lenz))
    for i in range(lenz):
        for j in range(lenz):
            if i != j:
                err[i, j] = np.sum((data[i]-data[j])**2)
                err[i, j] /= lenx*leny
    return err

def first_selection(data, faktor=5):
    """print('Deleted image ' + str(i) + ' with mpb ' + str(mpb[i]))
    A first selection to filter out empty classes and beamedge images.

    Parameters
    ----------
    data : Array 3D
        Input data.
    faktor : Integer (optional)
        Faktor by which the MSE must be off for it to be ruled out.

    Returns
    -------
    new_data : Array 3D
        This is the filtered data.
    del_data : Array 3D
        This is the bad data.

    """
    print('Making initial selection')
    lenz, leny, lenx = np.shape(data)
    mpb = meanpixelbrightness(data)
    delete_list = []
    delete_list2 = []
    for i in range(lenz):
        if mpb[i] == 0:
            delete_list.append(i)
    garbage = len(delete_list)
    print(f'{garbage} images deleted because of empty classes.')
    new_data = np.zeros((lenz-garbage, leny, lenx))
    del_data = np.zeros((garbage, leny, lenx))
    counter = 0
    counter2 = 0
    for j in range(lenz):
        if j not in delete_list:
            new_data[counter, :, :] = data[j, :, :]
            counter += 1
        else:
            del_data[counter2, :, :] = data[j, :, :]
            counter2 += 1

    mse = y_projections(MSE(new_data))
    for i in range(lenz-garbage):
        if mse[i] > np.sum(mse)/lenz*faktor:
            delete_list2.append(i)
            print('Deleted image ' + str(i) + ' with mpb ' +
                  str(mpb[i]) + ' and mse ' + str(mse[i]))
    garbage2 = len(delete_list2)
    new_data2 = np.zeros((lenz-garbage-garbage2, leny, lenx))
    del_data2 = np.zeros((garbage + garbage2, leny, lenx))
    counter = 0
    counter2 = 0
    for j in range(lenz-garbage):
        if j not in delete_list2:
            new_data2[counter, :, :] = new_data[j, :, :]
            counter += 1
        else:
            del_data2[counter2, :, :] = new_data[j, :, :]
            counter2 += 1

    return new_data2, del_data2