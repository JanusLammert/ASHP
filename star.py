#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:26:23 2023

@author: janus
"""

import starfile
import pandas as pd
import numpy as np

path="/home/janus/Jureca/230814_red_tau/Select/job068/particles.star"

star = starfile.read(path)


optics = pd.DataFrame.from_dict(star["optics"])
particles = pd.DataFrame.from_dict(star["particles"])

Micrographs = []

for index, row in particles.iterrows():
    Micrograph = row['rlnMicrographName']
    
    if Micrograph not in Micrographs:
        Micrographs.append(Micrograph)

ID = list(range(len(Micrographs)))

#%%

df = pd.DataFrame(columns=Micrographs)

df.loc[1] = [[] for _ in range(len(Micrographs))]
df.loc[2] = [0 for _ in range(len(Micrographs))]
df.loc[3] = [0 for _ in range(len(Micrographs))]

#%%

for index, row in particles.iterrows():
    Micrograph = row['rlnMicrographName']
    HelixID = row['rlnHelicalTubeID']
    
    if HelixID not in df.loc[1,Micrograph]:
        df.loc[1,Micrograph].append(HelixID)

for Micrograph in Micrographs:
    df.loc[2,Micrograph]=len(df.loc[1,Micrograph])
    
temp = 0
for Micrograph in Micrographs:
    df.loc[3,Micrograph]=df.loc[2,Micrograph]+temp
    temp += df.loc[2,Micrograph]

rows, col = particles.shape
new_indices = np.zeros(rows)


for index, row in particles.iterrows():
    Micrograph = row['rlnMicrographName']
    HelixID = row['rlnHelicalTubeID']
    
    new_indices[index] = 1 + df.loc[3, Micrograph] - df.loc[2, Micrograph] + df.loc[1, Micrograph].index(HelixID)
        
number_helices = np.max(new_indices)
number_classes = particles['rlnClassNumber'].max()

particles = particles.assign(rlnNewIndices=new_indices)

Matrix = np.zeros((int(number_classes), int(number_helices)))  
        
for index, row in particles.iterrows():
    ClassID = row['rlnClassNumber']
    HelixID = row['rlnNewIndices']
    
    Matrix[int(ClassID-1),int(HelixID-1)] +=1
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# =============================================================================
# temp=Micrographs[0]
# temp2=0
# 
# for i in Micrographs:
#     Micrographs_dict_sum[i] = len(Micrographs_dict[i])
# 
# for i in Micrographs[1:]:
#     Micrographs_dict_abs[i] = Micrographs_dict_abs[temp]+Micrographs_dict_sum[i]
#     temp = i
#     
# #del ID, Micrographs
# print(particles["rlnHelicalTubeID"].describe())
# =============================================================================
