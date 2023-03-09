#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:32:51 2022

@author: janus
"""

from sklearn import cluster as sklc

def Spectral_Clustering(Data, nCluster=3):
    clustering = sklc.SpectralClustering(n_clusters = nCluster, affinity='precomputed', assign_labels='discretize', random_state=0).fit(Data)
    clusters = clustering.labels_
    return clusters

def kmeans_Clustering(Data, nCluster=3):
    clustering = sklc.KMeans(n_clusters = nCluster, random_state=0).fit(Data)
    clusters = clustering.labels_
    return clusters

def DBscan_Clustering(Data, eps=0.5):
    clustering = sklc.DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit(Data)
    clusters = clustering.labels_
    return clusters

def Optics_Clustering(Data):
    clustering = sklc.OPTICS(metric="precomputed").fit(Data)
    clusters = clustering.labels_
    return clusters
