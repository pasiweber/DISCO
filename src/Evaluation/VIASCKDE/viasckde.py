import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.neighbors import KernelDensity
import seaborn as sns
import numpy as np

#############################################################################
#                                                                           #
#   Code from https://github.com/senolali/VIASCKDE/blob/main/VIASCKDE.py    #
#                                                                           #
#############################################################################

def closest_node(n, v):
    kdtree = KDTree(v)
    d, i = kdtree.query(n)
    return d


def viasckde_score(X, labels, krnl='gaussian', b_width=0.05):
    CoSeD = np.array([], [])
    num_k = np.unique(labels)
    kde = KernelDensity(kernel=krnl, bandwidth=b_width).fit(X)
    iso = kde.score_samples(X)

    ASC = np.array([])
    numC = np.array([])
    CoSeD = np.array([])
    viasc = 0
    if len(num_k) > 1:
        for i in num_k:
            data_of_cluster = X[labels == i]
            data_of_not_its = X[labels != i]
            isos = iso[labels == i]
            isos = (isos - min(isos)) / (max(isos) - min(isos))
            for j in range(len(data_of_cluster)):  # for each data of cluster j
                row = np.delete(data_of_cluster, j, 0)  # exclude the data j
                XX = data_of_cluster[j]
                a = closest_node(XX, row)
                b = closest_node(XX, data_of_not_its)
                ASC = np.hstack((ASC, ((b - a) / max(a, b)) * isos[j]))
            numC = np.hstack((numC, ASC.size))
            CoSeD = np.hstack((CoSeD, ASC.mean()))
        for k in range(len(numC)):
            viasc += numC[k] * CoSeD[k]
        viasc = viasc / sum(numC)
        print("viasc=%0.4f" % viasc)
    else:
        viasc = float("nan")
    return viasc