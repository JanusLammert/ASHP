#!/usr/bin/env python3
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

start = time.time()

ASHP_VERSION = "0.9"
progname = os.path.basename(sys.argv[0])
parser = ArgumentParser(description="ASHP: Autoselection of Helical Polymers. Tool to differetiate between different polymorphs based on the results of 2D-Classification.")
parser.add_argument("-i", "--infile", type=str, required=True, help="Pass the 2D classification Relion .mrc file")
parser.add_argument("-c", "--cluster_num", type=int, default=2, help="Number of clusters for clustering algorithm")
parser.add_argument("-a", "--algorithm", type=str, default="Spectral_clustering", help="Algorithm used for clustering. Options: Spectral_clustering, kmeans")
parser.add_argument("-o", "--outfile", type=str, default="result", help="Name of the outfile")
parser.add_argument("-m", "--metric", type=str, default="MSE", help="Metric for the distance parameter. Options: MSE, conv")
parser.add_argument("-w", "--window_size", type=int, default=35, help="Windowsize")
parser.add_argument("-n", "--num_window", type=int, default=30, help="Number of windows")
parser.add_argument("-t", "--topnum", type=int, default=5, help='Number of windows which get taken into account for classification')
parser.add_argument("-p", "--parallelise", type=int, default=-1, help='Number of cores')
parser.add_argument("-l", "--laplacian", action="store_true", help="Should the laplacian be calculated?")
parser.add_argument("-b", "--bar", action="store_true", help="Show progressbar")
parser.add_argument("-f", "--func", action="store_true", help="Toggle window-function")
parser.add_argument("-s", "--size", type=int, default=60, help="Sicereduction in ydirection to reduce computetime")
parser.add_argument("-r", "--reduction", type=int, default=5, help="Reduction of windowsize in comparison to class image")
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
    OSS.save_np(f.MSE_with_window(data, windows), name=options.outfile, spec='', path='') # + ".npy")
elif options.metric == "conv":
    OSS.save_np(f.Data_from_convMPPB(data, windows), name=options.outfile, spec='', path='') # + ".npy")
elif options.metric == "CC":
    OSS.save_np(f.Data_from_CC(data, windows), name=options.outfile, spec='', path='') # + ".npy")

end = time.time()

print(end-start)
