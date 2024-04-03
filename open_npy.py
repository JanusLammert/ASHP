#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:19:44 2022

@author: Janus Lammert ☺
"""

import OSS
import numpy as np
import Affinity_matrix as am
import clustering as c
import Preprocessing as p

input_data = '/home/janus/Uni/temp/Results/result_20230914-143027.npy'
input_img = '/home/janus/Uni/temp/class_averages.mrcs'

input_array = OSS.open_np(input_data)
data, bad = p.first_selection(OSS.open_mrc(input_img))
del bad
#OSS.show_img(data, save=False, path='Images/run_it025_classes_02', normalized=False)

dims = np.shape(input_array)

# =============================================================================
# for i in range(10):
#     for j in range(10):
#         OSS.Display_conv_data(input_array[i,j,:,:], i, j)
# =============================================================================
        


res = am.aff_matrix(input_array, mode='MSE')
#clusters9 = c.Spectral_Clustering(res, nCluster=9)
#clusters8 = c.Spectral_Clustering(res, nCluster=8)
#clusters7 = c.Spectral_Clustering(res, nCluster=7)
clusters6 = c.Spectral_Clustering(res, nCluster=6)
clusters5 = c.kmeans_Clustering(res, nCluster=5)
clusters4 = c.Spectral_Clustering(res, nCluster=4)
clusters3 = c.Spectral_Clustering(res, nCluster=3)
clusters2 = c.Spectral_Clustering(res, nCluster=2)


OSS.display_classes2(data, clusters6, path ="/home/janus/Uni/temp/Results/", name='MSE_run_it025_classes_02_clus6', save=True)
OSS.display_classes2(data, clusters5, path ="/home/janus/Uni/temp/Results/", name='MSE_run_it025_classes_02_clus5', save=True)
OSS.display_classes2(data, clusters4, path ="/home/janus/Uni/temp/Results/", name='MSE_run_it025_classes_02_clus4', save=True)
OSS.display_classes2(data, clusters3, path ="/home/janus/Uni/temp/Results/", name='MSE_run_it025_classes_02_clus3', save=True)
OSS.display_classes2(data, clusters2, path ="/home/janus/Uni/temp/Results/", name='MSE_run_it025_classes_02_clus2', save=True)


