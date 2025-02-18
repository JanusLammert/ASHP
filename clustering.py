#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:32:51 2022

@author: janus
"""

from sklearn import cluster as sklc
import Kruskal_clustering as kc

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

def Kruskal_based_clustering(Data, n_clusters=None, threshold=None):
    cluster_his, linkage = kc.Kruskal_clustering(Data)
    kc.plot_dendrogram(linkage)

    if n_clusters!=None and threshold!=None:
        print("Please only suplly only n_clusters or threshold!")
        return

    elif n_clusters != None:
        try:
            clusters = [cluster_his[(-1*n_clusters)]]
            return kc.rename_clusters(clusters)
        except Exception:
            print("n_clusters must be an integer")
            return
    
    elif threshold != None:
        try:
            for i in range(len(linkage)):
                if linkage[i,2] > threshold:
                    n_clusters=i
                    break
            clusters = [cluster_his[(-1*n_clusters)]]
            return kc.rename_clusters(clusters)
            print()
        except Exception:
            print("UUUUUUUPPPSSSS")
            return