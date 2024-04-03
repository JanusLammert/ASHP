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
    elif arg.algorithm=="DBscan":
        dis = am.dis_matrix(input_array, mode=arg.metric, num_points=arg.top_num)
        clusters = c.DBscan_Clustering(dis, eps=200)
        n_clus = np.max(clusters)+1
    elif arg.algorithm=="OPTICS":
        dis = am.dis_matrix(input_array, mode=arg.metric, num_points=arg.top_num)
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
fa.visualize_affinity_matrix_2(res)



