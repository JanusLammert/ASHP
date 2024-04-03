#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri MAR 03 10:30:44 2023

@author: Janus Lammert ☺
"""

import OSS
import numpy as np

infile='/home/janus/Uni/Masterarbeit/ASHP/data/run_conv_l_w20_n20_t_.npy'

input_array = OSS.open_np(infile)
dims = np.shape(input_array)

num_class_1 = 21
num_class_2 = 20
num_class_3 = 16
num_class_4 = 30
num_class_5 = 24

for i in [1,5,10,30, 31]:
    for j in [2,3,7,8]:
        OSS.Display_conv_data(input_array[i,j], i, j, save=True,path='Images/', name='convl_class_'+str(i)+'_to_'+str(j)+'.svg')