# Implementation of MMJ-SC and MMJ-CH by
# - Author: Gangli Liu - Github user `mike-liuliu`
# - Source: https://github.com/mike-liuliu/Min-Max-Jump-distance
# - License: Apache License Version 2.0, January 2004 (https://github.com/mike-liuliu/Min-Max-Jump-distance/blob/main/LICENSE.txt)

# Paper: Min-Max-Jump distance and its applications
# Authors: Gangli Liu
# Link: https://arxiv.org/abs/2301.05994


from sklearn.metrics import pairwise_distances
import numpy as np
import networkx as nx
import sys
from sklearn import metrics


class construct_MST_prim:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = None

    # A utility function to print
    # the constructed MST stored in parent[]
    def printMST(self, parent):
        #         print("Edge \tWeight")

        MST = []
        for i in range(1, self.V):
            #             print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
            MST.append([parent[i], i, self.graph[i][parent[i]]])
        return MST

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initialize min value
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1  # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):

                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        MST_list = self.printMST(parent)

        return MST_list


def construct_MST_from_graph(distance_matrix):

    lenX = len(distance_matrix)
    g = construct_MST_prim(lenX)
    g.graph = distance_matrix
    MST_list = g.primMST()

    MST = nx.Graph()
    for i in range(lenX):
        MST.add_node(i)
    for edge in MST_list:
        MST.add_edge(edge[0], edge[1], weight=edge[2])
    return MST


def cal_mmj_matrix_by_algo_4_Calculation_and_Copy(X, round_n=15):

    lenX = len(X)
    distance_matrix = pairwise_distances(X)
    distance_matrix = np.round(distance_matrix, round_n)
    mmj_matrix = np.zeros((lenX, lenX))

    MST = construct_MST_from_graph(distance_matrix)

    MST_edge_list = list(MST.edges(data="weight"))

    edge_node_list = [(edge[0], edge[1]) for edge in MST_edge_list]
    edge_weight_list = [edge[2] for edge in MST_edge_list]
    edge_large_to_small_arg = np.argsort(edge_weight_list)[::-1]
    edge_weight_large_to_small = np.sort(edge_weight_list)[::-1]
    edge_nodes_large_to_small = [edge_node_list[i] for i in edge_large_to_small_arg]

    for i, edge_nodes in enumerate(edge_nodes_large_to_small):
        edge_weight = edge_weight_large_to_small[i]
        MST.remove_edge(*edge_nodes)
        tree1_nodes = list(nx.dfs_preorder_nodes(MST, source=edge_nodes[0]))
        tree2_nodes = list(nx.dfs_preorder_nodes(MST, source=edge_nodes[1]))
        for p1 in tree1_nodes:
            for p2 in tree2_nodes:
                mmj_matrix[p1, p2] = mmj_matrix[p2, p1] = edge_weight

    return mmj_matrix


### --- mmj_ch_score --- ###


def cal_centroid_X(X_id, dis_matrix):

    distance_matrix_square = dis_matrix**2

    square_dis_list = [sum(distance_matrix_square[pp, X_id]) for pp in X_id]
    center_idx = np.argmin(square_dis_list)

    return center_idx


def cal_centroid_id(X, labels, mmj_matrix):
    #     import pdb;pdb.set_trace()
    n_labels = len(set(labels))

    X_id = np.array(range(len(X)))

    distance_matrix_square = mmj_matrix**2
    center_idx = []

    for kkk in range(n_labels):

        clu_index = X_id[labels == kkk]
        square_dis_list = [sum(distance_matrix_square[pp, clu_index]) for pp in clu_index]

        mmm = clu_index[np.argmin(square_dis_list)]
        center_idx.append(mmm)

    return center_idx


def mmj_ch_score(X, labels, ignor_less_than_n=1):
    dis_matrix = cal_mmj_matrix_by_algo_4_Calculation_and_Copy(X)

    n_samples = len(X)
    n_labels = len(set(labels))

    X_id = np.array(range(len(X)))

    extra_disp, intra_disp = 0.0, 0.0
    mean = cal_centroid_X(X_id, dis_matrix)

    centroids = cal_centroid_id(X, labels, dis_matrix)

    for k in range(n_labels):
        cluster_k = X_id[labels == k]

        # if a cluster contains less than ignor_less_than_n points, we just return the worst  calinski harabasz index score.
        if len(cluster_k) < ignor_less_than_n:
            return -np.inf

        mean_k = centroids[k]
        extra_disp += len(cluster_k) * (dis_matrix[mean, mean_k] ** 2)
        #         import pdb;pdb.set_trace()
        intra_disp += np.sum(dis_matrix[cluster_k, mean_k] ** 2)

    return 1.0 if intra_disp == 0.0 else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))


### --- mmj_db_score --- ###


def pairwise_dist_from_id(X_id, centroid_id, mmj_matrix):
    m, n = len(X_id), len(centroid_id)
    dists = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            p = X_id[i]
            q = centroid_id[j]
            dists[i, j] = mmj_matrix[p, q]

    return dists


def mmj_db_score(X, labels):
    dis_matrix = cal_mmj_matrix_by_algo_4_Calculation_and_Copy(X)

    n_labels = len(set(labels))

    X_id = np.array(range(len(X)))

    intra_dists = np.zeros(n_labels)
    centroids = cal_centroid_id(X, labels, dis_matrix)

    for k in range(n_labels):
        cluster_k = X_id[labels == k]

        centroid = centroids[k]

        temp_dis = pairwise_dist_from_id(cluster_k, [centroid], dis_matrix)

        intra_dists[k] = np.average(temp_dis)

    centroid_distances = pairwise_dist_from_id(centroids, centroids, dis_matrix)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0
    #     import pdb;pdb.set_trace()
    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists

    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)


### --- mmj_sc_score --- ###


def mmj_sc_score(X, labels, use_scikit=True):
    dis_matrix = cal_mmj_matrix_by_algo_4_Calculation_and_Copy(X)

    if use_scikit:
        return metrics.silhouette_score(dis_matrix, labels, metric="precomputed")
