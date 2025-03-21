#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:12:21 2022

@author: Janus Lammert ☺
"""

#spectral clustering

import functions as f
import OSS
import Preprocessing as p
import reshaping as r
import os
from argparse import ArgumentParser
import sys
import time
import numpy as np
import analyse_functions as anaf
import starfile
import multiprocessing as mp
import saving_star_2 as svs
import clustering as c
import Affinity_matrix as am
import star2 as chep

start = time.time()

ASHP_VERSION = "0.9"
progname = os.path.basename(sys.argv[0])
parser = ArgumentParser(description="ASHP: Autoselection of Helical Polymers. Tool to differetiate between different polymorphs based on the results of 2D-Classification.")
parser.add_argument("-i", "--infile", type=str, required=True, help="Pass the 2D classification Relion .mrc file")
parser.add_argument("-is", "--infile_star", type=str, required=True, help="Path to the particle or data starfile.")
parser.add_argument("-c", "--cluster_num", nargs='+', type=int, default=2, help="Number of clusters for clustering algorithm")
parser.add_argument("-a", "--algorithm", type=str, default="Spectral_clustering", help="Algorithm used for clustering. Options: Spectral_clustering, kmeans")
parser.add_argument("-og", "--outpath_graphs", type=str, default="result", help="Name of the outputpath for images")
parser.add_argument("-o", "--output", type=str, required=True, help="Path for the output of the starfile")
parser.add_argument("-m", "--metric", type=str, default="MSE", help="Metric for the distance parameter. Options: MSE, conv")
parser.add_argument("-w", "--window_size", type=int, default=35, help="Windowsize")
parser.add_argument("-n", "--num_window", type=int, default=30, help="Number of windows")
parser.add_argument("-t", "--topnum", type=int, default=5, help='Number of windows which get taken into account for classification')
parser.add_argument("-p", "--parallelise", type=int, default=-1, help='Number of cores')
parser.add_argument("-l", "--laplacian", action="store_true", help="Should the laplacian be calculated?")
parser.add_argument("-b", "--bar", action="store_true", help="Show progressbar")
parser.add_argument("-fu", "--func", action="store_true", help="Toggle window-function")
parser.add_argument("-s", "--size", type=int, default=60, help="Sicereduction in ydirection to reduce computetime")
parser.add_argument("-r", "--reduction", type=int, default=5, help="Reduction of windowsize in comparison to class image")
parser.add_argument("-f", "--factor", type=int, default=1, help="Weight between CHEP and ASHP. Lower means ASHP is wheighted stronger, higher means CHEP is weighted stronger")
parser.add_argument("-V", "--version", action="version", version="ASHP version: " + ASHP_VERSION, help="Print ASHP version and exit")
options = parser.parse_args()

print(options)

# %% Open File

data = OSS.open_mrc(options.infile)
good, bad = p.first_selection(data)
del bad

#data = r.ReduceSize(good, red_y=options.size)
data = good
data = r.ReduceSize(data, red_x=10)

if options.laplacian:
    data = p.laplacian(data)

if options.func:
    windows = r.ReduceSize(r.Window(data, windowsize=options.window_size, num_win=options.num_window, WindowFunction='flattop'), red_y=options.reduction)
else:
    windows = r.ReduceSize(r.Window(data, windowsize=options.window_size, num_win=options.num_window, WindowFunction='flat'), red_y=options.reduction)

if options.metric == "MSE":
    input_array = f.MSE_with_window(data, windows)
elif options.metric == "conv":
    input_array = f.Data_from_convMPPB(data, windows)
elif options.metric == "CC":
    input_array = f.Data_from_CC(data, windows)

end = time.time()

print(end-start)

# %% CHEP

dims = np.shape(input_array)
class_sim = am.aff_matrix(input_array, mode=options.metric, num_points=options.topnum)

print('Affinitymatrix calculated succesfully!')

class_norm, helix_norm, particles, optics = chep.chep(path=options.infile_star)

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

translation_table = translationtable(particles)

aff_class = r.reduce_matrix(aff_class, translation_table)
helix_norm = r.delete_rows(helix_norm, translation_table)

chep_aff_c = np.copy(aff_class)

aff_class = aff_class*class_sim**options.factor

print('Aff_class calculated succesfully!')

num_class = np.shape(class_sim)[0]

queue = []
pool = mp.Pool()

for i in range(num_helix):
    queue.append([i, num_helix, helix_norm, options.factor])

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

for nc in options.cluster_num:
    class_clusters = c.Spectral_Clustering(aff_class, nCluster=int(nc))
    helix_clusters = c.Spectral_Clustering(aff_helix, nCluster=int(nc))

    nc = str(nc)

    print('Clustering done!')

    list_df_class = svs.split_dataframe_by_classes(particles, class_clusters, translation_table)

    for i, particles in enumerate(list_df_class):
        starfile.write(svs.dataframes_to_dict(optics, particles), options.output + '/class_' + str(nc) + '_sorted_by_class_' + str(i) + '.star')

    list_df_helix = svs.split_dataframe_by_fibrils(particles, helix_clusters)

    for i, particles in enumerate(list_df_helix):
        starfile.write(svs.dataframes_to_dict(optics, particles), options.output + '/class_' + str(nc) + '_sorted_by_fib_' + str(i) + '.star')
        
    print('Successfully saved star files!')
    print('Generate plots')

    path_graph = options.outpath_graphs

    anaf.plot_matrix(chep_aff_c, class_clusters, path=path_graph, name='CHEP_aff_c_' + nc)
    anaf.plot_matrix(class_sim, class_clusters, path=path_graph, name='ASHP_aff_c_' + nc)
    anaf.plot_matrix(chep_aff_c-class_sim, class_clusters, path=path_graph, name='CHEP-ASHP-diff_aff_c_' + nc, cmap='bwr')

    anaf.plot_image_histogram(chep_aff_c, path=path_graph, name='CHEP_aff_c_hist_' + nc)
    anaf.plot_image_histogram(class_sim, path=path_graph, name='ASHP_aff_c_hist_' + nc)
    anaf.plot_image_histogram_diff(chep_aff_c-class_sim, path=path_graph, name='CHEP-ASHP-diff_aff_c_hist_' + nc)

    anaf.plot_matrix(aff_class, class_clusters, path=path_graph, name='CHEP-ASHP-comb_c_' + nc)
    anaf.plot_image_histogram(aff_class, path=path_graph, name='CHEP-ASHP-comb_c_hist_' + nc)

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

    anaf.separate_and_save_images(options.infile, class_clusters, 'class-averages-by-clusters_' + nc)

print('Run successfull!')
print('Exiting normally')
