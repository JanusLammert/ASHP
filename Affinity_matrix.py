#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:26:34 2022

@author: janus
"""

import numpy as np

# WindowSource, ClassImg, Windowindex

def aff_matrix(data, mode='MSE', num_points=5):
    shape = np.shape(data)
    distance = np.zeros((shape[0],shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            values = data[i,j,:,0]
            if mode == 'MSE':
                minV = values[np.argsort(values)[:num_points]]
                distance[i,j] = np.sum(minV)
            elif mode == 'conv':
                maxV = values[np.argsort(values)[(-1)*num_points:]]
                distance[i,j] = np.sum(maxV)
            else:
                print('ERROR')
    similarity = np.exp((-1)*distance / distance.std())
    return similarity

def dis_matrix(data, mode='MSE', num_points=5):
    shape = np.shape(data)
    distance = np.zeros((shape[0],shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            values = data[i,j,:,0]
            if mode == 'MSE':
                minV = values[np.argsort(values)[:num_points]]
                distance[i,j] = np.sum(minV)
            elif mode == 'conv':
                maxV = values[np.argsort(values)[(-1)*num_points:]]
                distance[i,j] = np.sum(maxV)
            else:
                print('ERROR')
    return distance

