import pandas as pd
import os
import sys

parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)


from src.Cluster.DPC.cluster import DensityPeakCluster
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering, MeanShift, AgglomerativeClustering
from src.Evaluation.dcdistances.dctree import DCTree


def KDBSCAN(X, l):
    dctree = DCTree(X)
    


CLUSTER_ALGORITHMS = {
    "ground truth": lambda X, l: l,
    "HDBSCAN": lambda X, l: HDBSCAN().fit(X).labels_,
    "DPC": lambda X, l: DensityPeakCluster().fit(X).labels_,
    "KMeans": lambda X, l: KMeans(len(set(l))).fit(X).labels_,
    "SpectralClustering": lambda X, l: SpectralClustering(len(set(l))).fit(X).labels_,
    "MeanShift": lambda X, l: MeanShift().fit(X).labels_,
    "Agglomerative": lambda X, l: AgglomerativeClustering(len(set(l))).fit(X).labels_,
    "DBSCAN": lambda X, l: DBSCAN()
}

SELECTED_CLUSTER_ALGORITHMS = [
    "DPC",
]

CLUSTER_ABBREV = {
    "DPC": "DCP",
}
