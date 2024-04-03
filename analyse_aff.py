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
import force_atlas as fa
import matplotlib.pyplot as plt


parser = ArgumentParser(description="get statistics on quality of clustering.")
parser.add_argument("-i", "--infile", type=str, help="Pass the path to the .npy-file")
parser.add_argument("-n", "--num_clusters", type=int, default=5, help="Provide the number pf clusters")
parser.add_argument("-d", "--display", action="store_true", help="display classification results. (Requieres classimages)")
parser.add_argument("-c", "--classimages", type=str, help="Pass the path to the mrc-file of the classimages")
parser.add_argument("-m", "--metric", type=str, default="MSE", help="MSE or conv")
parser.add_argument("-a", "--algorithm", type=str, default="Spectral_Clustering", help="type of clustering algorithm")
parser.add_argument("-s", "--save_csv", type=str, help="If the affinity map should get saved to a csv give a path here.")
parser.add_argument("-t", "--top_num", type=int, default=5, help="Numbers of windows taken into account")
arg = parser.parse_args()

num_class_1 = 21
num_class_2 = 20
num_class_3 = 16
num_class_4 = 30
num_class_5 = 24

def plot_matrix(matrix):
    plt.imshow(matrix,vmax=np.max(matrix[matrix != 1]), cmap='viridis')
    plt.colorbar(label='Value')
    plt.axis('off')
    plt.show()

def plot_matrix2(matrix):
    combined_axis_descriptions = ['Polymorph 1'] * 21 + ['Polymorph 2'] * 20 + ['Polymorph 3'] * 16 + ['Polymorph 4'] * 30 + ['Polymorph 5'] * 24
    
    # Plot the matrix
    plt.imshow(matrix, vmax=np.max(matrix[matrix != 1]), cmap='viridis')
    plt.colorbar(label='Value')
    
    # Customize the x-axis labels
    plt.xticks([0, 21, 41, 57, 87, 111], ['Polymorph 1', 'Polymorph 2', 'Polymorph 3', 'Polymorph 4', 'Polymorph 5', ''])
    
    # Customize the y-axis labels
    plt.yticks([0, 21, 41, 57, 87, 111], ['Polymorph 1', 'Polymorph 2', 'Polymorph 3', 'Polymorph 4', 'Polymorph 5', ''], rotation=90)
    
    plt.savefig('analyse_aff_matrix.svg')

    plt.show()

def plot_matrix_log(matrix):
    # Check if the minimum value is positive or close to zero
    min_val = np.min(matrix)
    if min_val <= 0:
        min_val = np.abs(min_val) + 1e-10  # Add a small offset if the minimum value is non-positive

    # Apply logarithmic transformation to the data with the offset
    matrix_log = np.log(matrix + min_val)
    
    # Plot the transformed data
    plt.imshow(matrix_log, cmap='viridis')
    
    # Add colorbar with label and logarithmic scale
    cbar = plt.colorbar(label='Log(Value)')
    cbar.ax.set_yscale('log')
    
    # Hide axis labels and ticks
    plt.axis('off')
    
    # Show the plot
    plt.show()

def generate_array():
    array_part_0 = np.zeros(21, dtype=int)
    array_part_1 = np.ones(20, dtype=int)
    array_part_2 = np.full(16, 2, dtype=int)
    array_part_3 = np.full(30, 3, dtype=int)
    array_part_4 = np.full(24, 4, dtype=int)
    
    result_array = np.concatenate((array_part_0, array_part_1, array_part_2, array_part_3, array_part_4))
    return result_array

Benedikt = generate_array()

def calculate_affinity_statistics(class_assignment, affinity_matrix):
    num_classes = len(np.unique(class_assignment))
    class_affinity_sum = np.zeros(num_classes)
    class_point_count = np.zeros(num_classes)
    class_outer_affinity_sum = np.zeros(num_classes)
    class_outer_point_count = np.zeros(num_classes)

    for i in range(len(class_assignment)):
        class_index = class_assignment[i]
        class_affinity_sum[class_index] += np.sum(affinity_matrix[i][class_assignment == class_index])
        class_point_count[class_index] += np.sum(class_assignment == class_index)
        class_outer_affinity_sum[class_index] += np.sum(affinity_matrix[i][class_assignment != class_index])
        class_outer_point_count[class_index] += np.sum(class_assignment != class_index)

    class_avg_affinity = class_affinity_sum / class_point_count
    class_avg_outer_affinity = class_outer_affinity_sum / class_outer_point_count

    return class_avg_affinity, class_avg_outer_affinity

if arg.infile:
    input_array = OSS.open_np(arg.infile)
    dims = np.shape(input_array)
    if arg.algorithm=="Spectral_Clustering":
        res = am.aff_matrix(input_array, mode=arg.metric, num_points=arg.top_num)
        clusters = c.Spectral_Clustering(res, nCluster=arg.num_clusters)
        n_clus = arg.num_clusters
    elif arg.algorithm=="kmeans":
        res = am.aff_matrix(input_array, mode=arg.metric, num_points=arg.top_num)
        clusters = c.kmeans_Clustering(res, nCluster=arg.num_clusters)
        n_clus = arg.num_clusters
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

    print(" Class || Polymorph 1 | Polymorph 2 | Polymorph 3 | Polymorph 4 | Polymorph 5")
    print("=============================================================================")

    for i in range(arg.num_clusters):
        print(f"   {i+1}   ||      {tabel[0,i]}      |      {tabel[1,i]}      |      {tabel[2,i]}      |      {tabel[3,i]}      |      {tabel[4,i]}     ")
        print("-----------------------------------------------------------------------------")

if arg.classimages:
    data, bad = p.first_selection(OSS.open_mrc(arg.classimages))
    del bad

if arg.display:
    if arg.classimages:
        OSS.display_classes2(data, clusters6, name=f'MSE_run_it025_classes_02_clus{arg.num_clusters}', save=True)
    else:
        print("Please also supply classimages")

if arg.save_csv:
    np.savetxt("Affinitymap.csv", res, delimiter=',')

res = fa.sym(res)
#fa.visualize_affinity_matrix3(res, k=1)
print(np.min(res))
#plot_matrix_log(res)

in_class, out_class = calculate_affinity_statistics(clusters, res)

print(in_class/out_class)


plot_matrix2(res)



