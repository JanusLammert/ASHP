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
import saving_star_2 as svs
import reshaping as r
import multiprocessing as mp
import starfile
import analyse_functions as anaf

infile = "/p/scratch/cvsk18/lammert1/240403_ASHP_CHEP/ASHP_out.npy"
infile_star = "/p/scratch/cvsk18/lammert1/240403_ASHP_CHEP/particles.star"
num_clusters = 4
metric = "MSE"
f = 1
top_num = 5
path_graph = '/p/scratch/cvsk18/lammert1/240403_ASHP_CHEP/images/'
mrc_file='/p/scratch/cvsk18/lammert1/240403_ASHP_CHEP/class_averages.mrcs'
class_star = "/p/scratch/cvsk18/lammert1/240403_ASHP_CHEP/class_averages.star"

print('Parameters read in succesfully!')

input_array = OSS.open_np(infile)
dims = np.shape(input_array)
class_sim = am.aff_matrix(input_array, mode=metric, num_points=top_num)

print('Affinitymatrix calculated succesfully!')

class_norm, helix_norm, particles, optics = chep.chep(path=infile_star)

print('CHEP ran succesfully!')

num_class, num_helix = np.shape(class_norm)

aff_class = np.zeros((num_class, num_class))
aff_helix = np.zeros((num_helix, num_helix))

def multiply_two(class_sim, vec1, vec2, f):
    sim = np.sum(vec1[:, np.newaxis] * vec2 * class_sim ** f)
    return sim

def multiply_two_wrapper(start, num_helix, vec, f):
    results = np.zeros(num_helix)
    for j in range(start, num_helix):
        results[j] = multiply_two(class_sim, vec[:, start], vec[:,j], f)
    return start, results

def translationtable(df, column_name='rlnClassNumber'):
    unique_values = df[column_name].unique()
    unique_integers = np.asarray(sorted(set(int(value) for value in unique_values))) - 1
    return unique_integers

for i in range(num_class):
    for j in range(i,num_class):
        aff_class[i,j] = np.dot(class_norm[i,:], class_norm[j,:])
        aff_class[j,i] = aff_class [i,j]

translation_table = svs.read_class_averages_star(class_star)
print(translation_table)
print(translationtable(particles))

aff_class = r.reduce_matrix(aff_class, translation_table)
helix_norm = r.delete_rows(helix_norm, translation_table)

chep_aff_c = np.copy(aff_class)

aff_class = aff_class*class_sim**f

print('Aff_class calculated succesfully!')

num_class = np.shape(class_sim)[0]

queue = []
pool = mp.Pool()

for i in range(num_helix):
    queue.append([i, num_helix, helix_norm, f])

print("Starting multiprocessing!")

res = pool.starmap(multiply_two_wrapper, queue)
del queue

for element in res:
    aff_helix[:,element[0]] = element[1]

for i in range(num_helix):
    for j in range(i,num_helix):
        aff_helix[i, j] = aff_helix[j, i]

#print(aff_helix)

print('Aff_helix calculated succesfully!')

aff_class = aff_class/np.max(aff_class)
aff_helix = aff_helix/np.max(aff_helix)

print('Normalization done!')

class_clusters = c.Spectral_Clustering(aff_class, num_clusters)
helix_clusters = c.Spectral_Clustering(aff_helix, num_clusters)

print('Clustering done!')

list_df_class = svs.split_dataframe_by_classes(particles, class_clusters, translation_table)

for i, particles in enumerate(list_df_class):
    starfile.write(svs.dataframes_to_dict(optics, particles), '/p/scratch/cvsk18/lammert1/240403_ASHP_CHEP/2nd_class/class2_sorted_by_class_' + str(i) + '.star')

list_df_helix = svs.split_dataframe_by_fibrils(particles, helix_clusters)

for i, particles in enumerate(list_df_helix):
    starfile.write(svs.dataframes_to_dict(optics, particles), '/p/scratch/cvsk18/lammert1/240403_ASHP_CHEP/2nd_class/class2_sorted_by_fib_' + str(i) + '.star')
    
print('Successfully saved star files!')
print('Generate plots')

anaf.plot_matrix(chep_aff_c, class_clusters, path=path_graph, name='CHEP_aff_c')
anaf.plot_matrix(class_sim, class_clusters, path=path_graph, name='ASHP_aff_c')
anaf.plot_matrix(chep_aff_c-class_sim, class_clusters, path=path_graph, name='CHEP-ASHP-diff_aff_c', cmap='bwr')

anaf.plot_image_histogram(chep_aff_c, path=path_graph, name='CHEP_aff_c_hist')
anaf.plot_image_histogram(class_sim, path=path_graph, name='ASHP_aff_c_hist')
anaf.plot_image_histogram_diff(chep_aff_c-class_sim, path=path_graph, name='CHEP-ASHP-diff_aff_c_hist')

anaf.plot_matrix(aff_class, class_clusters, path=path_graph, name='CHEP-ASHP-comb_c')
anaf.plot_image_histogram(aff_class, path=path_graph, name='CHEP-ASHP-comb_c_hist')

inpoly, outpoly = anaf.calculate_affinity_statistics(class_clusters, chep_aff_c)
print()
print('CHEP:')
print(inpoly/outpoly)
inpoly, outpoly = anaf.calculate_affinity_statistics(class_clusters, class_sim)
print()
print('ASHP:')
print(inpoly/outpoly)
inpoly, outpoly = anaf.calculate_affinity_statistics(class_clusters, chep_aff_c-class_sim)
print()
print('CHEP-ASHP:')
print(inpoly/outpoly)
inpoly, outpoly = anaf.calculate_affinity_statistics(class_clusters, aff_class)
print()
print('CHEP+ASHP:')
print(inpoly/outpoly)

anaf.plot_matrix(aff_helix, helix_clusters, path=path_graph, name='CHEP-ASHP-comb_aff_h')
anaf.plot_image_histogram(aff_helix, path=path_graph, name='CHEP-ASHP-comb_aff_h_hist')
inpoly, outpoly = anaf.calculate_affinity_statistics(helix_clusters, aff_helix)
print()
print('CHEP+ASHP helix:')
print(inpoly/outpoly)

anaf.separate_and_save_images(mrc_file, class_clusters, 'class-averages-by-clusters')

print('Run successfull!')
print('Exiting normally')
