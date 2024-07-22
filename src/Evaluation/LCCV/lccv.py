from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial.distance import cdist
import numpy as np


#######################################################################################
#                                                                                     #
#   Code from https://github.com/adanjoga/cvik-toolbox/blob/master/cvi/lccvindex.m    #
#                                                                                     #
#######################################################################################
def nan_searching(X, dist):
    N = X.shape[0]
    r = 0
    flag = 0
    nb = np.zeros(N)
    nb_r = np.zeros(N)
    NNr = [[] for i in range(N)]
    RNNr = [[] for i in range(N)]
    LN = [[] for i in range(N)]
    count = []
    while flag == 0:
        for i in range(N):
            dists = dist[i, :]
            indices = np.argsort(dists)
            y = indices[r]
            nb[y] = nb[y] + 1
            NNr[i].append(y)
            RNNr[y].append(i)
        nb_r = nb
        count.append(list(nb_r).count(0))
        if count[-1] == 0 or count[-1] == count[-2]:
            flag = 1
        r = r + 1
    lamd = r - 1
    for i in range(N):
        LN[i] = NNr[nb_r[i]]
    return lamd, LN


def local_density(LN, dist, i):
    # LN local neighbors as indices
    # dist distance matrix
    # i point
    mu = len(LN[i])
    dists = [dist[i, j] for j in range(LN[i])]
    return mu / sum(dists)


def LORE(LN, rho, X, dist):
    rep = [[] for i in range(len(X))]
    local_cores = []

    for i in range(len(X)):
        maxdens = 0
        max_index = 0
        for j in LN[i]:
            if maxdens < rho[j]:
                maxdens = rho[j]
                max_index = j
        for p in LN[i]:
            if len(rep[p]) == 0:
                rep[p] = [max_index]
            elif len(rep[p]) != 0 and rep[p][0] != max_index:
                if dist[p][max_index] < dist[p][rep[p][0]]:
                    rep[p] = [max_index]
            for z in range(len(X)):
                if rep[z][0] == p:
                    rep[z] = [max_index]

    for i in range(len(X)):
        if rep[i][0] == i:
            local_cores.append(i)
    return local_cores, rep


def lccv_score(X, labels):
    dist = cdist(X, X)
    N = len(dist)
    lamd, LN = nan_searching(X, dist)
    rho = [local_density(LN, dist, i) for i in range(N)]
    local_cores, rep = LORE(LN, rho, X, dist)
    rep_count = [
        x
        for xs in rep
        for x in xs
    ]
    # Designated saturated neighbor graph
    # if j is one of the lamd nearest neighbors there exists an edge
    conn = np.ones(N, N)
    conn = conn * np.inf
    for i in range(N):
        for j in LN[i]:
            conn[i, j] = dist[i, j]
    graph = csr_matrix(conn)
    dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
    local_cores_in = {key: [] for key in np.unique(labels)}
    for i in local_cores:
        local_cores_in[labels[i]].append(i)
    lccv_sum = 0
    for i in local_cores:
        label = labels[i]
        local_core_in_A = local_cores_in[label]
        if len(local_core_in_A) == 1:
            lccv_sum += 0
        else:
            n_l_A = len(local_core_in_A)
            dists = [dist_matrix[i, j] for j in local_core_in_A]
            a_i = (1 / (n_l_A - 1)) * np.sum(dists)
            cluster_wise_dists = []
            for l in np.unique(labels):
                if l != label:
                    dists = [dist_matrix[i, j] for j in local_cores_in[l]]
                    cluster_wise_dists.append((1 / len(local_cores_in[l])) * sum(dists))
            b_i = min(cluster_wise_dists)
            lccv = sil_eq(a_i, b_i)
            n_i = rep_count.count(i)
            lccv_sum = lccv_sum + lccv * n_i
    lccv_c = (1 / len(local_cores)) * lccv_sum
    return lccv_c


def sil_eq(a, b):
    return (b - a) / np.max(a, b)
