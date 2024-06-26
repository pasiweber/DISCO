import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
# changed imports
from hdbscan._hdbscan_linkage import mst_linkage_core
from hdbscan.hdbscan_ import isclose
#from hdbscan.hdbscan_ import isclose, kruskal_mst_with_mutual_reachability
from scipy.sparse.csgraph import minimum_spanning_tree
def all_points_core_distance(distance_matrix, d=2.0):
    """
    Compute the all-points-core-distance for all the points of a cluster.

    Parameters
    ----------
    distance_matrix : array (cluster_size, cluster_size)
        The pairwise distance matrix between points in the cluster.

    d : integer
        The dimension of the data set, which is used in the computation
        of the all-point-core-distance as per the paper.

    Returns
    -------
    core_distances : array (cluster_size,)
        The all-points-core-distance of each point in the cluster

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    distance_matrix[distance_matrix != 0] = (1.0 / distance_matrix[
        distance_matrix != 0]) ** d
    result = distance_matrix.sum(axis=1)
    result /= distance_matrix.shape[0] - 1

    if result.sum() == 0:
        result = np.zeros(len(distance_matrix))
    else:
        result **= (-1.0 / d)

    return result


def max_ratio(stacked_distances):
    max_ratio = 0
    for i in range(stacked_distances.shape[0]):
        for j in range(stacked_distances.shape[1]):
            dist = stacked_distances[i][j][0]
            coredist = stacked_distances[i][j][1]
            if dist == 0 or coredist/dist <= max_ratio:
                continue
            max_ratio = coredist/dist

    return max_ratio


def distances_between_points(X, labels, cluster_id,
                                    metric='sqeuclidean', d=None, no_coredist=False,
                                    print_max_raw_to_coredist_ratio=False, **kwd_args):
    """
    Compute pairwise distances for all the points of a cluster.

    If metric is 'precomputed' then assume X is a distance matrix for the full
    dataset. Note that in this case you must pass in 'd' the dimension of the
    dataset.

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.

    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    cluster_id : integer
        The cluster label for which to compute the distances

    metric : string
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.

    d : integer (or None)
        The number of features (dimension) of the dataset. This need only
        be set in the case of metric being set to `precomputed`, where
        the ambient dimension of the data is unknown to the function.

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------

    distances : array (n_samples, n_samples)
        The distances between all points in `X` with `label` equal to `cluster_id`.

    core_distances : array (n_samples,)
        The all-points-core_distance of all points in `X` with `label` equal
        to `cluster_id`.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    if metric == 'precomputed':
        if d is None:
            raise ValueError('If metric is precomputed a '
                             'd value must be provided!')
        distance_matrix = X[labels == cluster_id, :][:, labels == cluster_id]
    else:
        subset_X = X[labels == cluster_id, :]
        distance_matrix = pairwise_distances(subset_X, metric=metric,
                                             **kwd_args)
        ## check here
        d = X.shape[1]

    if no_coredist:
        return distance_matrix, None

    else:
        core_distances = all_points_core_distance(distance_matrix.copy(), d=d)
        core_dist_matrix = np.tile(core_distances, (core_distances.shape[0], 1))
        stacked_distances = np.dstack(
            [distance_matrix, core_dist_matrix, core_dist_matrix.T])

        if print_max_raw_to_coredist_ratio:
            print("Max raw distance to coredistance ratio: " + str(max_ratio(stacked_distances)))

        return stacked_distances.max(axis=-1), core_distances


def internal_minimum_spanning_tree(mr_distances, algorithm):
    """
    Compute the 'internal' minimum spanning tree given a matrix of mutual
    reachability distances. Given a minimum spanning tree the 'internal'
    graph is the subgraph induced by vertices of degree greater than one.

    Parameters
    ----------
    mr_distances : array (cluster_size, cluster_size)
        The pairwise mutual reachability distances, inferred to be the edge
        weights of a complete graph. Since MSTs are computed per cluster
        this is the all-points-mutual-reachability for points within a single
        cluster.
    algorithm: string (default = 'prim')
        which algo for the SMT

    Returns
    -------
    internal_nodes : array
        An array listing the indices of the internal nodes of the MST

    internal_edges : array (?, 3)
        An array of internal edges in weighted edge list format; that is
        an edge is an array of length three listing the two vertices
        forming the edge and weight of the edge.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """


    min_span_tree = None

    if 'prim_claudius' in algorithm:
        min_span_tree = prims_mst_with_mutual_reachability(mr_distances)
    elif 'prim_official' in algorithm:
        min_span_tree = matlab_prims_wrapper(mr_distances)
    elif 'prim_lena' in algorithm:
        min_span_tree = internal_minimal_spanning_tree_Prim(mr_distances)
    elif 'prim2_lena' in algorithm:
        min_span_tree = prim_mst_with_mutual_reachability(mr_distances)
    elif 'prim_hdbscan' in algorithm:
        single_linkage_data = mst_linkage_core(mr_distances)
        min_span_tree = single_linkage_data.copy()
        for index, row in enumerate(min_span_tree[1:], 1):
            candidates = np.where(isclose(mr_distances[int(row[1])], row[2]))[0]
            candidates = np.intersect1d(candidates,
                                        single_linkage_data[:index, :2].astype(
                                            int))
            candidates = candidates[candidates != row[1]]
            assert len(candidates) > 0
            row[0] = candidates[0]
    elif 'kruskal_lena' in algorithm:
        min_span_tree = kruskal2_mst_with_mutual_reachability(mr_distances)
    elif 'kruskal_claudius' in algorithm:
        min_span_tree = kruskal_mst_with_mutual_reachability(mr_distances)

    vertices = np.arange(mr_distances.shape[0])[
        np.bincount(min_span_tree.T[:2].flatten().astype(np.intp)) > 1]
    if not len(vertices):
        vertices = [0]
    # A little "fancy" we select from the flattened array reshape back
    # (Fortran format to get indexing right) and take the product to do an and
    # then convert back to boolean type.
    edge_selection = np.prod(np.in1d(min_span_tree.T[:2], vertices).reshape(
        (min_span_tree.shape[0], 2), order='F'), axis=1).astype(bool)

    # Density sparseness is not well defined if there are no
    # internal edges (as per the referenced paper). However
    # MATLAB code from the original authors simply selects the
    # largest of *all* the edges in the case that there are
    # no internal edges, so we do the same here
    if np.any(edge_selection):
        # If there are any internal edges, then subselect them out
        edges = min_span_tree[edge_selection]
    else:
        # If there are no internal edges then we want to take the
        # max over all the edges that exist in the MST, so we simply
        # do nothing and return all the edges in the MST.
        edges = min_span_tree.copy()

    return vertices, edges, len(vertices)


def density_separation(X, labels, cluster_id1, cluster_id2,
                       internal_nodes1, internal_nodes2,
                       core_distances1, core_distances2,
                       metric='sqeuclidean', no_coredist=False, **kwd_args):
    """
    Compute the density separation between two clusters. This is the minimum
    distance between pairs of points, one from internal nodes of MSTs of each cluster.

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.

    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    cluster_id1 : integer
        The first cluster label to compute separation between.

    cluster_id2 : integer
        The second cluster label to compute separation between.

    internal_nodes1 : array
        The vertices of the MST for `cluster_id1` that were internal vertices.

    internal_nodes2 : array
        The vertices of the MST for `cluster_id2` that were internal vertices.

    core_distances1 : array (size of cluster_id1,)
        The all-points-core_distances of all points in the cluster
        specified by cluster_id1.

    core_distances2 : array (size of cluster_id2,)
        The all-points-core_distances of all points in the cluster
        specified by cluster_id2.

    metric : string
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------
    The 'density separation' between the clusters specified by
    `cluster_id1` and `cluster_id2`.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    if metric == 'precomputed':
        sub_select = X[labels == cluster_id1, :][:, labels == cluster_id2]
        distance_matrix = sub_select[internal_nodes1, :][:, internal_nodes2]
    else:
        cluster1 = X[labels == cluster_id1][internal_nodes1]
        cluster2 = X[labels == cluster_id2][internal_nodes2]
        distance_matrix = cdist(cluster1, cluster2, metric, **kwd_args)

    if no_coredist:
        return distance_matrix.min()

    else:
        core_dist_matrix1 = np.tile(core_distances1[internal_nodes1],
                                    (distance_matrix.shape[1], 1)).T
        core_dist_matrix2 = np.tile(core_distances2[internal_nodes2],
                                    (distance_matrix.shape[0], 1))

        mr_dist_matrix = np.dstack([distance_matrix,
                                    core_dist_matrix1,
                                    core_dist_matrix2]).max(axis=-1)

        return mr_dist_matrix.min()


def validity_index(X, labels, metric='sqeuclidean',
                    d=None, per_cluster_scores=False, mst_raw_dist=False, verbose=False, algorithm='prim',  **kwd_args):
    """
    Compute the density based cluster validity index for the
    clustering specified by `labels` and for each cluster in `labels`.

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.

    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    metric : optional, string (default 'euclidean')
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.

    d : optional, integer (or None) (default None)
        The number of features (dimension) of the dataset. This need only
        be set in the case of metric being set to `precomputed`, where
        the ambient dimension of the data is unknown to the function.

    per_cluster_scores : optional, boolean (default False)
        Whether to return the validity index for individual clusters.
        Defaults to False with the function returning a single float
        value for the whole clustering.

    mst_raw_dist : optional, boolean (default False)
        If True, the MST's are constructed solely via 'raw' distances (depending on the given metric, e.g. euclidean distances)
        instead of using mutual reachability distances. Thus setting this parameter to True avoids using 'all-points-core-distances' at all.
        This is advantageous specifically in the case of elongated clusters that lie in close proximity to each other <citation needed>.


    algorithm: optional, string (default 'prim')
        Which algo for the MS

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------
    validity_index : float
        The density based cluster validity index for the clustering. This
        is a numeric value between -1 and 1, with higher values indicating
        a 'better' clustering.

    per_cluster_validity_index : array (n_clusters,)
        The cluster validity index of each individual cluster as an array.
        The overall validity index is the weighted average of these values.
        Only returned if per_cluster_scores is set to True.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    # cluster labels
    cluster_labels = np.unique(labels)

    for i in range(len(cluster_labels)):
        if np.sum(labels == cluster_labels[i]) == 1:
            labels[labels == cluster_labels[i]] = -1
            cluster_labels[i] = -1

    clusters = np.setdiff1d(cluster_labels, -1)
    if len(clusters) == 0 or len(clusters) == 1:
        return 0,0
    O = X.shape[0]
    X = X[labels != -1, :]
    labels = labels[labels != -1]

    cluster_labels, counts = np.unique(labels, return_counts=True)

    core_distances = {}
    density_sparseness = {}
    mst_nodes = {}
    mst_edges = {}

    max_cluster_id = labels.max() + 1
    max_cluster_id = len(cluster_labels)
    density_sep = np.inf * np.ones((max_cluster_id, max_cluster_id),
                                   dtype=np.float64)
    cluster_validity_indices = np.empty(max_cluster_id, dtype=np.float64)
    internal_mst_per_cluster = {}
    for cluster_id in cluster_labels:
        if np.sum(labels == cluster_id) == 0:
            continue

        distances_for_mst, core_distances[
            cluster_id] = distances_between_points(
            X,
            labels,
            cluster_id,
            metric,
            d,
            no_coredist=mst_raw_dist,
            print_max_raw_to_coredist_ratio=verbose,
            **kwd_args
        )

        mst_nodes[cluster_id], mst_edges[cluster_id], len_internal_mst = \
            internal_minimum_spanning_tree(distances_for_mst, algorithm)
        density_sparseness[cluster_id] = mst_edges[cluster_id].T[2].max()
        internal_mst_per_cluster['Number_Edges_Internal_{}'.format(cluster_id)] = len_internal_mst
        internal_mst_per_cluster['Density_Sparseness_{}'.format(cluster_id)] = density_sparseness[cluster_id]

    for i in range(max_cluster_id):

        if np.sum(labels == i) == 0:
            continue

        internal_nodes_i = mst_nodes[i]
        for j in range(i + 1, max_cluster_id):

            if np.sum(labels == j) == 0:
                continue

            internal_nodes_j = mst_nodes[j]
            density_sep[i, j] = density_separation(
                X, labels, i, j,
                internal_nodes_i, internal_nodes_j,
                core_distances[i], core_distances[j],
                metric=metric, no_coredist=mst_raw_dist,
                **kwd_args
            )
            density_sep[j, i] = density_sep[i, j]




    n_samples = float(X.shape[0])
    n_samples = O
    result = 0

    for i in range(max_cluster_id):

        if np.sum(labels == i) == 0:
            continue

        min_density_sep = density_sep[i].min()
        cluster_validity_indices[i] = (
            (min_density_sep - density_sparseness[i]) /
            max(min_density_sep, density_sparseness[i])
        )

        if verbose:
            print("Minimum density separation: " + str(min_density_sep))
            print("Density sparseness: " + str(density_sparseness[i]))
        internal_mst_per_cluster['Density_Separation_{}'.format(i)] = min_density_sep

        cluster_size = np.sum(labels == i)
        result += (cluster_size / n_samples) * cluster_validity_indices[i]

    if per_cluster_scores:
        return result, cluster_validity_indices, internal_mst_per_cluster
    else:
        return result, internal_mst_per_cluster

def kruskal_mst_with_mutual_reachability(mutual_reachability):
    n_vertices = mutual_reachability.shape[0]
    edges = [(i, j, mutual_reachability[i, j]) for i in range(n_vertices) for j in range(i + 1, n_vertices) if mutual_reachability[i, j] != np.inf]
    edges.sort(key=lambda x: x[2])
    parent = [i for i in range(n_vertices)]
    rank = [0 for _ in range(n_vertices)]
    mst_edges = []
    for edge in edges:
        u, v, w = edge
        uroot = find(parent, u)
        vroot = find(parent, v)
        if uroot != vroot:
            mst_edges.append(edge)
            union(parent, rank, uroot, vroot)
            if len(mst_edges) == n_vertices - 1:
                break
    # Format the MST for direct use in post-processing
    mst_formatted = np.zeros((len(mst_edges), 3))
    for i, (u, v, w) in enumerate(mst_edges):
        mst_formatted[i] = [u, v, w]
    return mst_formatted
def prim_mst_with_mutual_reachability(mutual_reachability):

    # Prims algorithm
    n_vertices = mutual_reachability.shape[0]
    G = {
        "no_vertices": n_vertices,
        "MST_edges": np.zeros((n_vertices - 1, 3)),
        "MST_degrees": np.zeros((n_vertices), dtype=int),
        "MST_parent": np.zeros((n_vertices), dtype=int),
    }

    # Prims algorithm
    d = []
    # 0 if not visited 1 if visited
    intree = []
    # initialize
    for i in range(G["no_vertices"]):
            intree.append(0)
            d.append(np.inf)
            G["MST_parent"][i] = int(i)
    start =0
    d[start] = 0
    all_nodes = [i for i in range(G["no_vertices"])]
    visited = []
    visited = np.array(visited).astype(int)
    v = start
    counter = -1
    # count until we connected all nodes
    while len(visited) != G["no_vertices"]-2:
    #while len(visited) != 2:
            counter = counter + 1
            # we add v to the 'visited'
            intree[v] = 1
            # remove it from allnodes
            all_nodes.remove(v)
            # add to visited
            visited = np.append(visited, [v])

            # we look only at edges that are outgoing vom visited vertices
            edges = mutual_reachability[visited, :]

            # edges going to already visited nodes are set to infinity
            edges[:, visited] = np.inf

            # we want to look at the smallest
            min_index = np.where(edges == edges.min())
            #print('Edges')
            #print(edges)
            #print('Min')
            #print(edges.min())
            #print('Min Index')
            #print(min_index)


            # we extract indices
            parent, v = visited[min_index[0]], min_index[1]
            parent = parent[0]
            v = v[0]
            if v in visited:
                print('Something went wrong')
            d[v] = edges.min()
            G["MST_parent"][v] = int(parent)
            G["MST_edges"][counter, :] = [
                parent,
                v,
                mutual_reachability[parent, v],
            ]
            G["MST_degrees"][parent] = G["MST_degrees"][parent] + 1
            G["MST_degrees"][v] = G["MST_degrees"][v] + 1



    # Format the MST for direct use in post-processing
    mst_formatted = np.zeros((len(G["MST_edges"]), 3))
    for i, (u, v, w) in enumerate(G["MST_edges"]):
        mst_formatted[i] = [u, v, w]
    return mst_formatted
def internal_minimal_spanning_tree_Prim(mrd):
    """
    Compute the minimal spanning tree and exclude external nodes and edges.
    External nodes are those with a degree of less than two.

    Args:
        mrd: Mutual reachability distance as list of lists

    Returns:
        mst_tmp: internal minimal spanning tree if existing
    """
    # transform to array
    mrd = np.array(mrd)
    intree = [0]*mrd.shape[0]
    parent = {}
    d = [np.inf]*mrd.shape[0]
    # calculate minimal spanning tree and extract adjacency matrix
    mst_temp = np.zeros([mrd.shape[0], mrd.shape[1]])
    d[0] = 0
    vertex = 0
    G = {"MST_edges": np.zeros((mrd.shape[0] - 1, 3)),}
    counter = -1
    while counter < mrd.shape[0]-2:
        intree[vertex] = 1
        dist = np.inf
        for neighbor in range(mrd.shape[0]):
            if(vertex!= neighbor) & (intree[neighbor] == 0):
                weight = mrd[vertex, neighbor]
                if d[neighbor]> weight:
                    d[neighbor] = weight
                    parent[str(neighbor)] = vertex
                if dist > d[neighbor]:
                    dist = d[neighbor]
                    next_v = neighbor
        counter = counter +1
        outgoing = parent[str(next_v)]
        incoming = next_v
        mst_temp[outgoing, incoming] = mrd[outgoing, incoming]

        G["MST_edges"][counter, :] = [
            outgoing,
            incoming,
            mrd[outgoing, incoming],
        ]
        mst_temp[incoming, outgoing] = mrd[outgoing, incoming]
        vertex = next_v

    # degrees of vertices (necessary to exclude external nodes)
    degrees = np.count_nonzero(mst_temp, axis=1)
    # indices of external nodes
    external_nodes = np.asarray(degrees < 2).nonzero()[0]
    # if we have only external nodes / no internal nodes we take the full MST
    if len(mrd) - len(external_nodes) > 1:
        # set edges from and to external nodes to 0
        mst_temp[external_nodes] = 0
        mst_temp[:, external_nodes] = 0
    result = []
    vertices = np.asarray(degrees>1).nonzero()[0]
    for i in range(len(mst_temp)):
        for j in range(i+1,len(mst_temp[i])):
            if mst_temp[i,j]!=0:
                result.append([i, j, mst_temp[i,j]])
    mst_formatted = np.zeros((len(G["MST_edges"]), 3))
    for i, (u, v, w) in enumerate(G["MST_edges"]):
        mst_formatted[i] = [u, v, w]
    return mst_formatted

def kruskal2_mst_with_mutual_reachability(mutual_reachability):

    # Prims algorithm
    n_vertices = mutual_reachability.shape[0]
    G = {
        "no_vertices": n_vertices,
        "MST_edges": np.zeros((n_vertices - 1, 3)),
        "MST_degrees": np.zeros((n_vertices), dtype=int),
        "MST_parent": np.zeros((n_vertices), dtype=int),
    }
    # transform to array
    mrd = np.array(mutual_reachability)
    # calculate minimal spanning tree and extract adjacency matrix
    # this calculates Kruskal
    mst = minimum_spanning_tree(mrd).toarray()

    counter = 0
    for outgoing in range(mst.shape[0]):
        for incoming in range(mst.shape[1]):
            if mst[outgoing, incoming]!=0:
                G["MST_edges"][counter, :] = [
                    outgoing,
                    incoming,
                    mst[outgoing, incoming],
                ]
                counter = counter +1

    # Format the MST for direct use in post-processing
    mst_formatted = np.zeros((len(G["MST_edges"]), 3))
    for i, (u, v, w) in enumerate(G["MST_edges"]):
        mst_formatted[i] = [u, v, w]
    return mst_formatted

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def prims_mst_with_mutual_reachability(mutual_reachability):
    #print('actually prims')
    n_vertices = mutual_reachability.shape[0]
    visited = set([0])  # Start from vertex 0
    edge_queue = []

    # Populate the initial edge queue with edges from vertex 0
    for j in range(1, n_vertices):
        if mutual_reachability[0, j] != np.inf:
            heapq.heappush(edge_queue, (mutual_reachability[0, j], 0, j))

    mst_edges = []

    while len(mst_edges) < n_vertices - 1 and edge_queue:
        weight, u, v = heapq.heappop(edge_queue)
        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, weight))

            # Add new edges from the newly added vertex
            for j in range(n_vertices):
                if mutual_reachability[v, j] != np.inf and j not in visited:
                    heapq.heappush(edge_queue, (mutual_reachability[v, j], v, j))

    # Format the MST for direct use in post-processing
    mst_formatted = np.zeros((len(mst_edges), 3))
    for i, (u, v, w) in enumerate(mst_edges):
        mst_formatted[i] = [u, v, w]

    return mst_formatted
import heapq


def matlab_prims_wrapper(mutual_reachability):
    n_vertices = mutual_reachability.shape[0]
    start = 0  # Starting vertex for Prim's algorithm

    # Initialize the graph structure expected by the provided Prim's algorithm
    G = {
        "no_vertices": n_vertices,
        "MST_parent": np.zeros(n_vertices, dtype=int),
        "MST_edges": np.zeros((n_vertices - 1, 3)),
        "MST_degrees": np.zeros(n_vertices, dtype=int)
    }

    # Replace np.inf with a large number for algorithm compatibility
    max_weight = np.max(mutual_reachability[np.isfinite(mutual_reachability)]) + 1
    G_edges_weights = np.where(mutual_reachability == np.inf, max_weight, mutual_reachability)

    # Call the provided MST_Edges function with the prepared data
    mst_edges, _ = MST_Edges(G, start, G_edges_weights)
    return mst_edges



def MST_Edges(G, start, G_edges_weights):

    # Prims algorithm
    d = []
    intree = []
    # initialize
    for i in range(G["no_vertices"]):
        intree.append(0)
        d.append(np.inf)
        G["MST_parent"][i] = int(i)

    d[start] = 0
    v = start
    counter = -1
    # count until we connected all nodes
    while counter < G["no_vertices"] - 2:
        # we add v to the 'visited'
        intree[v] = 1
        dist = np.inf
        # for every node
        for w in range(G["no_vertices"]):
            # if the node is not already in the visited elements and is not the same as the one we want to check
            if (w != v) & (intree[w] == 0):
                # we look at the distance
                weight = G_edges_weights[v, w]
                # if the distance is smaller than the distance that connects w currently we update
                if d[w] > weight:
                    d[w] = weight
                    G["MST_parent"][w] = int(v)
                # if the distance is smaller than current dist we update dist
                if dist > d[w]:
                    dist = d[w]
                    next_v = w
        counter = counter + 1
        outgoing = G["MST_parent"][next_v]
        incoming = next_v
        G["MST_edges"][counter, :] = [
            outgoing,
            incoming,
            G_edges_weights[outgoing, incoming],
        ]
        G["MST_degrees"][G["MST_parent"][next_v]] = (
            G["MST_degrees"][G["MST_parent"][next_v]] + 1
        )
        G["MST_degrees"][next_v] = G["MST_degrees"][next_v] + 1
        v = next_v
    Edg = G["MST_edges"]
    Degr = G["MST_degrees"]
    return Edg, Degr
