#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:14:32 2022

@author: Janus Lammert ☺

Functions
---------
Faltungen : calculates convolutions and saves them as an image. (not recommended)

Data_from_conv : retreves the point of the highest value and its value from a convolution (single process)

Data_from_convMPPB : retreves the point of the highest value and its value from a convolution (multi process)

MSE_with_window : retreves the point and its value with the lowest difference apart from 0 from a MSE calculation

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing as mp
import time as t

# Print iterations progress

if __name__ == "__main__":
    print("execute main.py")

def Faltungen(class_img, class_window):
    """
    GIB MIR MEMORY!!! GROMPF GROMPF GROMPF

    Parameters
    ----------
    class_img : 3D Array
        Klassenbilder.
    class_window : 4D Array
        Windows.

    Returns
    -------
    results : 5D Array
        Faltungen.

    """
    lenz_img, leny_img, lenx_img = np.shape(class_img)
    ind_1, ind_2, dy, dx = np.shape(class_window)
    results = np.zeros((ind_1, ind_2, lenz_img, leny_img, lenx_img))
    for i in range(ind_1):
        for j in range(ind_2):
            for k in range(lenz_img):
                if k != i:
                    results[i, j, k, :, :] = sp.signal.fftconvolve(
                        class_img[k, :, :], class_window[i, j, :, :], mode='same')
    return results


def conv(ind_2, class_img, class_window):
    """
    Internal function to calculate the maximum value and its position.

    Parameters
    ----------
    ind_2 : Integer
        amount of Windows.
    class_img : Array 2D
        Image that gets convoluted with the windows.
    class_window : Array 3D
        Stacked window images.

    Returns
    -------
    results_new : Array 2D
        First index for the Window that got used. 2nd Index: 0 = max Value; 1 = ypos; 2 = xpos

    """
    results_new = np.zeros((ind_2, 3))
    for j in range(ind_2):
        temp = sp.signal.fftconvolve(class_img, class_window[j, :, :])
        max_v = np.max(temp)
        result = np.where(temp == np.max(temp))
        results_new[j, :] = np.array([max_v, result[0], result[1]])
    return results_new


def Data_from_conv(class_img, class_window):
    """
    Retrives the maximum value and its position from a convolution between class 
    images and the windows of the other classes. \n
    WARNING: Only makes use of one Process. For multiprocessing use Data_from_convMPPB()

    Parameters
    ----------
    class_img : 3D Array
        Images of the of the classes stacked.
    class_window : 4D Array
        Array with the Windows of the different classes. Can take the output of the Window function.

    Returns
    -------
    results : 4D Array
        1st index: Class that the window is comming from.\n
        2nd index: Class the window got applied to.\n
        3rd index: Index of the window.\n
        4th index: 0: max Value; 1: y_pos; 2: x_pos.

    """
    lenz_img, leny_img, lenx_img = np.shape(class_img)
    ind_1, ind_2, dy, dx = np.shape(class_window)
    results = np.zeros((ind_1, lenz_img, ind_2, 3))
    ti = t.time()
    for i in tqdm.tqdm(range(ind_1), desc='Calculating convolutions:'):
        for k in range(lenz_img):
            if i != k:
                results[i, k, :, :] = conv(
                    ind_2, class_img[k, :, :], class_window[i, :, :, :])
    tf = t.time()
    print(f'total singel: {tf-ti}s')
    return results


def convMP(ind_2, i, k, class_img, class_window):
    """
    Internal function to calculate the maximum value and its position.

    Parameters
    ----------
    ind_2 : Integer
        amount of Windows.
    i : Integer
        Class that the window is comming from.
    k : Integer
        Class the window got applied to.
    class_img : Array 2D
        Image that gets convoluted with the windows.
    class_window : Array 3D
        Stacked window images.

    Returns
    -------
    i : Integer
        Class that the window is comming from.
    k : Integer
        Class the window got applied to.
    results_new : Array 2D
        1st index: Index of the window.\n
        2nd index: 0: max Value; 1: y_pos; 2: x_pos.

    """
    results_new = np.zeros((ind_2, 3, 4))
    res = np.zeros((ind_2, 3))
    mirrored = np.fliplr(class_img)
    rot = np.rot90(class_img,2)
    rot_mir = np.fliplr(class_img)
    c_img = [class_img, mirrored, rot, rot_mir]
    for l in range(4):
        for j in range(ind_2):
            temp = sp.signal.fftconvolve(c_img[0], class_window[j, :, :])
            max_v = np.max(temp)
            result = np.where(temp == np.max(temp))
            results_new[j, :, l] = np.array(
                [max_v, result[0], result[1]], dtype=float)
    for j in range(ind_2):
# =============================================================================
#         ind = np.where(results_new[j,0,:] == np.max(results_new[j,0,:]))
#         res[j,:] = results_new[j,:,ind[0]]
# =============================================================================
        cache = results_new[j,:,0]
#        index = 0
        for m in range(4):
            if results_new[j,0,m] > cache[0]:
#                index = m
                cache = results_new[j,:,m]
        res[j,:] = cache
        print(res)
    return i, k, res

def CC(ind_2, i, k, class_img, class_window):
    """
    Internal function to calculate the maximum value and its position.

    Parameters
    ----------
    ind_2 : Integer
        amount of Windows.
    i : Integer
        Class that the window is comming from.
    k : Integer
        Class the window got applied to.
    class_img : Array 2D
        Image that gets convoluted with the windows.
    class_window : Array 3D
        Stacked window images.

    Returns
    -------
    i : Integer
        Class that the window is comming from.
    k : Integer
        Class the window got applied to.
    results_new : Array 2D
        1st index: Index of the window.\n
        2nd index: 0: max Value; 1: y_pos; 2: x_pos.

    """
    results_new = np.zeros((ind_2, 3, 4))
    res = np.zeros((ind_2, 3))
    mirrored = np.fliplr(class_img)
    rot = np.rot90(class_img,2)
    rot_mir = np.fliplr(class_img)
    c_img = [class_img, mirrored, rot, rot_mir]
    for l in range(4):
        for j in range(ind_2):
            temp = sp.signal.correlate(c_img[0], class_window[j, :, :])
            max_v = np.max(temp)
            result = np.where(temp == np.max(temp))
            results_new[j, :, l] = np.array(
                [max_v, result[0], result[1]], dtype=float)
    for j in range(ind_2):
        cache = results_new[j,:,0]
        for m in range(4):
            if results_new[j,0,m] > cache[0]:
                cache = results_new[j,:,m]
        res[j,:] = cache
    return i, k, res

def Data_from_convMP(class_img, class_window):
    """
    Retrives the maximum value and its position from a convolution between class 
    images and the windows of the other classes. This version makes use of Multiprocessing. 
    It uses the maximum number of available cores. For singel process version use
    Data_from_conv().

    Parameters
    ----------
    class_img : 3D Array
        Images of the of the classes stacked.
    class_window : 4D Array
        Array with the Windows of the different classes. Can take the output of the Window function.

    Returns
    -------
    results : 4D Array
        1st index: Class that the window is comming from.\n
        2nd index: Class the window got applied to.\n
        3rd index: Index of the window.\n
        4th index: 0: max Value; 1: y_pos; 2: x_pos.

    """
    print('Calculating convolutions.')
    lenz_img, leny_img, lenx_img = np.shape(class_img)
    ind_1, ind_2, dy, dx = np.shape(class_window)
    queue = []
    pool = mp.Pool()
    for i in range(ind_1):
        for k in range(lenz_img):
            if i != k:
                queue.append([ind_2, i, k, class_img[k, :, :],
                             class_window[i, :, :, :]])
    res = pool.starmap(convMP, queue)
    del queue
    results = np.zeros((ind_1, lenz_img, ind_2, 3))
    for i in range(len(res)):
        results[res[i][0], res[i][1], :, :] = res[i][2]
    return results


def Data_from_convMPPB(class_img, class_window):
    """
    Retrives the maximum value and its position from a convolution between class 
    images and the windows of the other classes. This version makes use of Multiprocessing. 
    It uses the maximum number of available cores. Also it includes a progressbar. 
    For singel process version use Data_from_conv().

    Parameters
    ----------
    class_img : 3D Array
        Images of the of the classes stacked.
    class_window : 4D Array
        Array with the Windows of the different classes. Can take the output of the Window function.

    Returns
    -------
    results : 4D Array
        1st index: Class that the window is comming from.\n
        2nd index: Class the window got applied to.\n
        3rd index: Index of the window.\n
        4th index: 0: max Value; 1: y_pos; 2: x_pos.

    """
    lenz_img, leny_img, lenx_img = np.shape(class_img)
    ind_1, ind_2, dy, dx = np.shape(class_window)
    queue = []
    pool = mp.Pool()
    for i in range(ind_1):
        for k in range(lenz_img):
            if i != k:
                queue.append([ind_2, i, k, class_img[k, :, :],
                             class_window[i, :, :, :]])
    res = pool.starmap(convMP, queue)
    del queue
    results = np.zeros((ind_1, lenz_img, ind_2, 3))
    for i in range(len(res)):
        results[res[i][0], res[i][1], :, :] = res[i][2]
    return results

def Data_from_CC(class_img, class_window):
    """
    Retrives the maximum value and its position from a Crosscorrelation between class 
    images and the windows of the other classes. This version makes use of Multiprocessing. 
    It uses the maximum number of available cores. Also it includes a progressbar. 
    For singel process version use Data_from_conv().

    Parameters
    ----------
    class_img : 3D Array
        Images of the of the classes stacked.
    class_window : 4D Array
        Array with the Windows of the different classes. Can take the output of the Window function.

    Returns
    -------
    results : 4D Array
        1st index: Class that the window is comming from.\n
        2nd index: Class the window got applied to.\n
        3rd index: Index of the window.\n
        4th index: 0: max Value; 1: y_pos; 2: x_pos.

    """
    lenz_img, leny_img, lenx_img = np.shape(class_img)
    ind_1, ind_2, dy, dx = np.shape(class_window)
    queue = []
    pool = mp.Pool()
    for i in range(ind_1):
        for k in range(lenz_img):
            if i != k:
                queue.append([ind_2, i, k, class_img[k, :, :],
                             class_window[i, :, :, :]])
    res = pool.starmap(CC, queue)
    del queue
    results = np.zeros((ind_1, lenz_img, ind_2, 3))
    for i in range(len(res)):
        results[res[i][0], res[i][1], :, :] = res[i][2]
    return results

def inner_MSE_function(j, k, Data, Window):
    """
    inner function, that gets split up to all cores (SLOW use inner_MSE_function2)

    Parameters
    ----------
    j : integer
        index for sorting.
    k : integer
        index for sorting.
    Data : 2D Array
        Image that the Windows get applyed to.
    Window : 3D Array
        Stack of windows.

    Returns
    -------
    j : integer
        same as input.
    k : integer
        same as input.
    result : 2D Array
        1st index: Index of the window.\n
        2nd index: 0: max Value; 1: y_pos; 2: x_pos.

    """
    leny, lenx = np.shape(Data)
    wini, winy, winx = np.shape(Window)
    numy = leny-winy
    numx = lenx - winx
    results = np.zeros((wini, numy, numx))
    for y in range(numy):
        for x in range(numx):   
            for i in range(wini):
                for y2 in range(winy):
                    for x2 in range(winx):
                        results[i, y, x] += (Data[y+y2, x+x2] -
                                             Window[i, y2, x2])**2
    result = np.zeros((wini, 3))
    for i in range(wini):
        minV = np.min(results[i, :, :])
        pos = np.where(results[i, :, :] == minV)
        result[i, :] = np.array([minV, pos[0], pos[1]], dtype=float)
    return j, k, result


def inner_MSE_function2(j, k, Data, Window):
    """
    inner function, that gets split up to all cores

    Parameters
    ----------
    j : integer
        index for sorting.
    k : integer
        index for sorting.
    Data : 2D Array
        Image that the Windows get applyed to.
    Window : 3D Array
        Stack of windows.

    Returns
    -------
    j : integer
        same as input.
    k : integer
        same as input.
    result : 2D Array
        1st index: Index of the window.\n
        2nd index: 0: max Value; 1: y_pos; 2: x_pos.

    """
    leny, lenx = np.shape(Data)
    wini, winy, winx = np.shape(Window)
    numy = leny - winy
    numx = lenx - winx
    results = np.zeros((wini, numy, numx))
    result = np.zeros((wini, 3, 4))
    res = np.zeros((wini, 3))
    mirrored = np.fliplr(Data)
    rot = np.rot90(Data,2)
    rot_mir = np.fliplr(Data)
    c_img = [Data, mirrored, rot, rot_mir]
    for l in range(4):
        for y in range(numy):
            for x in range(numx):
                for i in range(wini):
                    results[i, y, x] = np.sum(np.square(np.subtract(
                        c_img[l][y:y+winy, x:x+winx], Window[i, :, :])))
                    if results[i, y, x] <= 0.1:
                        results[i, y, x] = 1000000000 #setzt alle Punkte mit dem Wert 0 auf 1Mrd da sonst flächen, die =0 sind auch eine MSD von 0 haben.
        for i in range(wini):
            minV = np.min(results[i, :, :])
            pos = np.where(results[i, :, :] == minV)
            print(minV, pos[0], pos[1])
            result[i, :,l] = np.array([minV, pos[0], pos[1]], dtype=float)
    for l in range(wini):
        cache = result[l,:,0]
#        index = 0
        for m in range(4):
            if result[l,0,m] > cache[0]:
#                index = m
                cache = result[l,:,m]
        res[l,:] = cache
    return j, k, res


def MSE_with_window(ClassImg, Windows):
    """
    Parameters
    ----------
    ClassImg : 3D-Array
        Images of the of the classes stacked.
    Windows : 4D Array
        Array with the Windows of the different classes. Can take the output of the Window function.

    Returns
    -------
    results : 4D Array
        1st index: Class that the window is comming from.\n
        2nd index: Class the window got applied to.\n
        3rd index: Index of the window.\n
        4th index: 0: min Value; 1: y_pos; 2: x_pos.

    """
    print('Calculating MSE.')
    print('Number of threads available: ' + str(mp.cpu_count()))
    lenz_img, leny_img, lenx_img = np.shape(ClassImg)
    ind_1, ind_2, dy, dx = np.shape(Windows)

    queue = []
    pool = mp.Pool()
    for i in range(ind_1):
        for k in range(lenz_img):
            if i != k:
                queue.append([i, k, ClassImg[k, :, :], Windows[i, :, :, :]])
    res = pool.starmap(inner_MSE_function2, queue)
    del queue
    results = np.zeros((ind_1, lenz_img, ind_2, 3))
    for i in range(len(res)):
        results[res[i][0], res[i][1], :, :] = res[i][2]
    return results
