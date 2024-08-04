import pandas as pd
import os
import sys

parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)


from src.Cluster.DPC.cluster import DensityPeakCluster
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering, MeanShift, AgglomerativeClustering
from src.Evaluation.dcdistances.dctree import DCTree
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
import numpy as np


def optimal_k_dbscan(X, l):
    dctree = DCTree(X, min_points=5, min_points_mr=5)

    l_ = np.full(len(l), -1)

    for k in range(2, len(set(l)) + 2):
        eps = dctree.get_eps_for_k(k)
        l_dbscan = DBSCAN(eps).fit(X).labels_
        l_kcenter = dctree.get_k_center(k)
        if ARI(l_dbscan, l_kcenter) < 0.98:
            break
        l_ = l_dbscan
    return l_


CLUSTER_ALGORITHMS = {
    "GroundTruth": lambda X, l: l,
    "OptimalKDBSCAN": lambda X, l: optimal_k_dbscan(X, l),
    "DBSCAN": lambda X, l: DBSCAN(DCTree(X).get_eps_for_k(len(set(l)))).fit(X).labels_,
    "KCenter": lambda X, l: DCTree(X).get_k_center(len(set(l))),
    "HDBSCAN": lambda X, l: HDBSCAN().fit(X).labels_,
    "DPC": lambda X, l: DensityPeakCluster().fit(X).labels_,
    "SpectralClustering": lambda X, l: SpectralClustering(len(set(l))).fit(X).labels_,
    "Agglomerative": lambda X, l: AgglomerativeClustering(len(set(l))).fit(X).labels_,
    "MeanShift": lambda X, l: MeanShift().fit(X).labels_,
    "KMeans": lambda X, l: KMeans(len(set(l))).fit(X).labels_,
}

# SELECTED_CLUSTER_ALGORITHMS = [
#     "DPC",
# ]

SELECTED_CLUSTER_ALGORITHMS = CLUSTER_ALGORITHMS.keys()

CLUSTER_ABBREV = {
    "GroundTruth": "GT",
    "OptimalKDBSCAN": "K'-DBSCAN",
    "DBSCAN": "K-DBSCAN",
    "KCenter": "KCenter",
    "HDBSCAN": "HDBSCAN",
    "DPC": "DPC",
    "SpectralClustering": "SC",
    "Agglomerative": "Aggl.",
    "MeanShift": "MeanShift",
    "KMeans": "KMeans",
}
