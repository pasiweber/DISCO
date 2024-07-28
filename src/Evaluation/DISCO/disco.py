import numpy as np
import functools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.neighbors import KDTree
from scipy.sparse import issparse

from src.Evaluation.dcdistances.dctree import DCTree


def disco_score(X: np.ndarray, labels: np.ndarray, min_points: int = 5) -> float:
    dc_distances = DCTree(X, min_points=min_points, no_fastindex=False).dc_distances()

    # Labels needs to be a one dimensional vector
    labels = np.reshape(labels, -1)

    disco_values = disco_samples(X, labels, min_points, dc_distances)
    noise_values = noise_samples(X, labels, min_points, dc_distances)

    return float(np.mean(np.concatenate((disco_values, noise_values))))


def disco_samples(
    X: np.ndarray,
    labels: np.ndarray,
    min_points: int = 5,
    dc_distances: np.ndarray | None = None,
) -> np.ndarray:
    if len(X) == 0:
        raise ValueError("Can't calculate DISCO score for empty dataset.")
    if len(X) != len(labels):
        raise ValueError("Dataset size differs from label size.")

    label_set = set(labels)
    # Only noise
    if label_set == {-1}:
        return np.array([])
    # One cluster without noise
    if len(label_set) == 1 and label_set != {-1}:
        return np.full(len(X), 0)
    # One cluster with noise
    if len(label_set) == 2 and -1 in label_set:
        labels = labels.copy()
        labels[labels == -1] = np.arange(-2, -len(labels[labels == -1]) - 2, -1)

    if dc_distances is None:
        dc_distances = DCTree(X, min_points=min_points, no_fastindex=False).dc_distances()

    # DISCO evaluation per non noise sample
    DISCO_evaluations = silhouette_samples(
        dc_distances[np.ix_(labels != -1, labels != -1)],
        labels[labels != -1],
        metric="precomputed",
    )
    return DISCO_evaluations


def noise_samples(
    X: np.ndarray,
    labels: np.ndarray,
    min_points: int = 5,
    dc_distances: np.ndarray | None = None,
) -> np.ndarray:
    if len(X) == 0:
        raise ValueError("Can't calculate noise score for empty dataset.")
    if len(X) != len(labels):
        raise ValueError("Dataset size differs from label size.")

    label_set = set(labels)
    # Only noise
    if label_set == {-1}:
        return np.full(len(X), -1)
    # No noise
    if -1 not in label_set:
        return np.array([])

    ## At least one cluster and noise ##
    if dc_distances is None:
        dc_distances = DCTree(X, min_points=min_points, no_fastindex=False).dc_distances()

    # Noise evaluation per noise sample
    tree = KDTree(X)
    core_dists, _ = tree.query(X, k=min_points)
    core_dists = core_dists.max(axis=1)

    # Get maximum core distance per cluster
    cluster_ids = set(labels[labels != -1])
    max_core_dist = np.empty(len(cluster_ids))
    for i, id in enumerate(cluster_ids):
        max_core_dist[i] = core_dists[labels == id].max()

    # Core property of noise evaluation
    noise_core_prop = np.full(len(labels[labels == -1]), np.inf)
    for i in range(len(cluster_ids)):
        noise_core_prop_i = (core_dists[labels == -1] - max_core_dist[i]) / np.maximum(
            core_dists[labels == -1], max_core_dist[i]
        )
        noise_core_prop = np.minimum(noise_core_prop, noise_core_prop_i)

    # Distance property of noise evaluation
    noise_dc_prop = np.full(len(labels[labels == -1]), np.inf)
    for i, id in enumerate(cluster_ids):
        min_dist_cluster_i = np.min(dc_distances[np.ix_(labels == -1, labels == id)], axis=1)
        noise_dc_i = (min_dist_cluster_i - max_core_dist[i]) / np.maximum(
            min_dist_cluster_i, max_core_dist[i]
        )
        noise_dc_prop = np.minimum(noise_dc_prop, noise_dc_i)

    # Noise evaluation is minimum of core and distance property
    return np.minimum(noise_core_prop, noise_dc_prop)


def silhouette_score(X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
    """Compute the mean Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels
    is ``2 <= n_labels <= n_samples - 1``.

    This function returns the mean Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`silhouette_samples`.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Read more in the :ref:`User Guide <silhouette_coefficient>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`~sklearn.metrics.pairwise_distances`. If ``X`` is
        the distance array itself, use ``metric="precomputed"``.

    sample_size : int, default=None
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for selecting a subset of samples.
        Used when ``sample_size is not None``.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.

    References
    ----------

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.metrics import silhouette_score
    >>> X, y = make_blobs(random_state=42)
    >>> kmeans = KMeans(n_clusters=2, random_state=42)
    >>> silhouette_score(X, kmeans.fit_predict(X))
    0.49...
    """
    if sample_size is not None:
        random_state = np.random.RandomState(1234)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))


def silhouette_samples(X, labels, *, metric="euclidean", **kwds):
    """Compute the Silhouette Coefficient for each sample.

    The Silhouette Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 ``<= n_labels <= n_samples - 1``.

    This function returns the Silhouette Coefficient for each sample.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.

    Read more in the :ref:`User Guide <silhouette_coefficient>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array. If
        a sparse matrix is provided, CSR format should be favoured avoiding
        an additional copy.

    labels : array-like of shape (n_samples,)
        Label values for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`~sklearn.metrics.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    silhouette : array-like of shape (n_samples,)
        Silhouette Coefficients for each sample.

    References
    ----------

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Examples
    --------
    >>> from sklearn.metrics import silhouette_samples
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>> X, y = make_blobs(n_samples=50, random_state=42)
    >>> kmeans = KMeans(n_clusters=3, random_state=42)
    >>> labels = kmeans.fit_predict(X)
    >>> silhouette_samples(X, labels)
    array([...])
    """

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == "precomputed":
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero "
            "elements on the diagonal. Use np.fill_diagonal(X, 0)."
        )
        if X.dtype.kind == "f":
            atol = np.finfo(X.dtype).eps * 100
            if np.any(np.abs(X.diagonal()) > atol):
                raise error_msg
        elif np.any(X.diagonal() != 0):  # integral dtype
            raise error_msg

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds["metric"] = metric
    reduce_func = functools.partial(_silhouette_reduce, labels=labels, label_freqs=label_freqs)
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)" % n_labels
        )


def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X.

    Parameters
    ----------
    D_chunk : {array-like, sparse matrix} of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk. If a sparse matrix is provided,
        only CSR format is accepted.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    n_chunk_samples = D_chunk.shape[0]
    # accumulate distances from each sample to each cluster
    cluster_distances = np.zeros((n_chunk_samples, len(label_freqs)), dtype=D_chunk.dtype)

    if issparse(D_chunk):
        if D_chunk.format != "csr":
            raise TypeError("Expected CSR matrix. Please pass sparse matrix in CSR format.")
        for i in range(n_chunk_samples):
            indptr = D_chunk.indptr
            indices = D_chunk.indices[indptr[i] : indptr[i + 1]]
            sample_weights = D_chunk.data[indptr[i] : indptr[i + 1]]
            sample_labels = np.take(labels, indices)
            cluster_distances[i] += np.bincount(
                sample_labels, weights=sample_weights, minlength=len(label_freqs)
            )
    else:
        for i in range(n_chunk_samples):
            sample_weights = D_chunk[i]
            sample_labels = labels
            cluster_distances[i] += np.bincount(
                sample_labels, weights=sample_weights, minlength=len(label_freqs)
            )

    # intra_index selects intra-cluster distances within cluster_distances
    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    # intra_cluster_distances are averaged over cluster size outside this function
    intra_cluster_distances = cluster_distances[intra_index]
    # of the remaining distances we normalise and extract the minimum
    cluster_distances[intra_index] = np.inf
    cluster_distances /= label_freqs
    inter_cluster_distances = cluster_distances.min(axis=1)
    return intra_cluster_distances, inter_cluster_distances
