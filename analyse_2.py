#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri MAR 03 10:30:44 2023

@author: Janus Lammert ☺
"""

import OSS
import numpy as np
import Affinity_matrix as am
import clustering as c
import Preprocessing as p
from argparse import ArgumentParser
from scipy.optimize import linear_sum_assignment

parser = ArgumentParser(description="get statistics on quality of clustering.")
parser.add_argument("-i", "--infile", type=str, help="Pass the path to the .npy-file")
parser.add_argument("-n", "--num_clusters", type=int, default=5, help="Provide the number pf clusters")
parser.add_argument("-m", "--metric", type=str, default="MSE", help="MSE or conv")
parser.add_argument("-a", "--algorithm", type=str, default="Spectral_Clustering", help="type of clustering algorithm")
parser.add_argument("-t", "--top_num", type=int, default=5, help="Numbers of windows taken into account")
arg = parser.parse_args()

def calculate_accuracy_unknown_order(confusion_matrix):
    """
    Calculate accuracy from a confusion matrix when the order of rows and columns is unknown.

    Parameters:
    confusion_matrix (list of lists): Confusion matrix where rows represent true labels and columns represent predicted labels.

    Returns:
    accuracy (float): Accuracy of the classification algorithm.
    """
    # Apply the Hungarian algorithm to find the optimal mapping
    true_indices, pred_indices = linear_sum_assignment(confusion_matrix, maximize=True)
    
    # Compute accuracy using the optimal mapping
    total_samples = np.sum(confusion_matrix)
    
    total_correct = 0
    for i in range(arg.num_clusters):
        total_correct+=confusion_matrix[true_indices[i], pred_indices[i]]

    if total_samples == 0:
        return 0.0
    
    accuracy = total_correct / total_samples
    return accuracy

num_class_1 = 21
num_class_2 = 20
num_class_3 = 16
num_class_4 = 30
num_class_5 = 24

if arg.metric == "MSD":
    mode="MSE"
elif arg.metric == "convolution":
    mode="conv"

if arg.infile:
    input_array = OSS.open_np(arg.infile)
    dims = np.shape(input_array)
    if arg.algorithm=="Spectral_Clustering":
        res = am.aff_matrix(input_array, mode=mode, num_points=arg.top_num)
        clusters = c.Spectral_Clustering(res, nCluster=arg.num_clusters)
        n_clus = arg.num_clusters
    elif arg.algorithm=="kmeans":
        res = am.aff_matrix(input_array, mode=mode, num_points=arg.top_num)
        clusters = c.kmeans_Clustering(res, nCluster=arg.num_clusters)
        n_clus = arg.num_clusters
    elif arg.algorithm=="DBscan":
        dis = am.dis_matrix(input_array, mode=mode, num_points=arg.top_num)
        clusters = c.DBscan_Clustering(dis, eps=200)
        n_clus = np.max(clusters)+1
    elif arg.algorithm=="OPTICS":
        dis = am.dis_matrix(input_array, mode=mode, num_points=arg.top_num)
        clusters = c.Optics_Clustering(dis)
        n_clus = np.max(clusters)+1
    else:
        print("Not a valid algorithm!")

    tabel = np.zeros((5, n_clus), dtype=np.int16)

    for i, v in enumerate(clusters):
        if i < num_class_1:
            tabel[0,v] += 1
        elif num_class_1 <= i and i < num_class_1 + num_class_2:
            tabel[1,v] += 1
        elif num_class_1 + num_class_2 <= i and i < num_class_1 + num_class_2 + num_class_3:
            tabel[2,v] += 1
        elif num_class_1 + num_class_2 + num_class_3 <= i and i < num_class_1 + num_class_2 + num_class_3 + num_class_4:
            tabel[3,v] += 1
        else:
            tabel[4,v] +=1

print(calculate_accuracy_unknown_order(tabel))







