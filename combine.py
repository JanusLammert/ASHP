#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:12:21 2022

@author: Janus Lammert ☺
"""

import star2 as chep
import OSS
import numpy as np
import Affinity_matrix as am
import clustering as c
import saving_star as svs

infile = ""
num_clusters = 5
metric = "MSE"
f = 1

input_array = OSS.open_np(infile)
dims = np.shape(input_array)
class_sim = am.aff_matrix(input_array, mode=metric)
#clusters = c.Spectral_Clustering(res, nCluster=num_clusters)

class_norm, helix_norm = chep.chep()

num_class, num_helix = np.shape(class_norm)

aff_class = np.zeros((num_class, num_class))
aff_helix = np.zeros((num_helix, num_helix))

def multiply_one(n_class, class_sim, vec1, vec2, f):
    sim = 0
    for i in range(n_class):
        for j in range(n_class):
            sim += vec1[i]*vec2[j]*class_sim[i,j]**f
    return sim

for i in range(num_class):
    for j in range(i,num_class):
        aff_class[i,j] = np.dot(class_norm[i,:], class_norm[j,:])
        aff_class[j,i] = aff_class [i,j]

aff_class = aff_class*class_sim**f

for i in range(num_helix):
    for j in range(i,num_helix):
        aff_helix[i,j] = multiply_one(num_class, class_sim, helix_norm[:,i], helix_norm[:,j], f)
        aff_helix[j,i] = aff_helix[i,j]

aff_class = aff_class/np.max(aff_class)
aff_helix = aff_helix/np.max(aff_helix)

class_clusters = c.Spectral_Clustering(aff_class, num_clusters)
helix_clusters = c.Spectral_Clustering(aff_helix, num_clusters)

