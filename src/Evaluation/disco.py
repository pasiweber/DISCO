import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import silhouette_samples
from src.Evaluation.dcdistances.dctree import DCTree


def disco_score(X: np.ndarray, labels: np.ndarray, min_points: int = 5):
    """Compute the mean DISCO score of all samples.

    The DISCO score is calculated using the mean intra-cluster on the
    dc-distance (``a``) and the mean nearest-cluster on the dc-distance (``b``)
    for each non noise sample.  The DISCO score for a non noise sample is 
    ``(b - a) / max(a, b)``.
    To clarify, ``b`` is the dc-distance between a non noise sample and the nearest
    cluster that the sample is not a part of.
    ``-1`` in labels are considered as noise and their DISCO score is calculated 
    by the minimum of the two different measures ``p_sparse`` and ``p_far``. 
    ``p_sparse`` measures how well the noise sample is within a sparse region.
    ``p_far`` measure how well the noise is remote to non noise samples.
    Note that DISCO score is defined for all possible number of labels
    ``1 <= n_labels <= n_samples``.
    Except for ``-1`` every other value is considered as cluster label.

    This function returns the mean DISCO score over all samples.
    To obtain the values for each sample, use :func:`disco_samples`.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Read more in the :ref:`User Guide <disco_score>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    disco : float
        Mean DISCO score for all samples.

    References
    ----------

    .. [1] `anonymous`_


    Examples
    --------
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.cluster import HDBSCAN
    >>> from disco import disco_score
    >>> X, y = make_moons(random_state=42)
    >>> hdbscan = HDBSCAN()
    >>> disco_score(X, hdbscan.fit_predict(X))
    0.71...
    """

    return np.mean(disco_samples(X, labels, min_points))


def disco_samples(X: np.ndarray, labels: np.ndarray, min_points: int = 5) -> np.ndarray:
    """Compute the DISCO score for each sample.

    The DISCO score is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    DISCO score are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.

    The DISCO score is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The DISCO score for a sample is ``(b - a) / max(a,
    b)``.
    Note that DISCO score is only defined if number of labels
    is 2 ``<= n_labels <= n_samples - 1``.

    This function returns the DISCO score for each sample.

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
        DISCO scores for each sample.

    References
    ----------

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the DISCO score
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

    if len(X) == 0:
        raise ValueError("Can't calculate DISCO score for empty dataset.")
    elif len(X) != len(labels):
        raise ValueError("Dataset size differs from label size.")

    # Labels needs to be a one dimensional vector
    labels = np.reshape(labels, -1)

    label_set = set(labels)
    # Only noise
    if label_set == {-1}:
        return np.full(len(X), -1)
    # One cluster without noise
    elif len(label_set) == 1 and label_set != {-1}:
        return np.full(len(X), 0)
    # One cluster with noise
    elif len(label_set) == 2 and -1 in label_set:
        dc_dists = DCTree(X, min_points=min_points, no_fastindex=False).dc_distances()
        l_ = labels.copy()
        l_[l_ == -1] = np.arange(-1, -len(l_[l_ == -1]) - 1, -1)
        disco_values = np.empty(len(X))
        disco_values[labels != -1] = p_cluster(dc_dists, l_, precomputed_dc_dists=True)[labels != -1]
        disco_values[labels == -1] = np.minimum(*p_noise(X, labels, min_points, dc_dists))
        return disco_values
    # More then one cluster with optional noise
    else:
        dc_dists = DCTree(X, min_points=min_points, no_fastindex=False).dc_distances()
        disco_values = np.empty(len(X))
        non_noise_dc_dists = dc_dists[np.ix_(labels != -1, labels != -1)]
        non_noise_labels = labels[labels != -1]
        disco_values[labels != -1] = p_cluster(non_noise_dc_dists, non_noise_labels, precomputed_dc_dists=True)
        disco_values[labels == -1] = np.minimum(*p_noise(X, labels, min_points, dc_dists))
        return disco_values


def p_cluster(
    X: np.ndarray,
    labels: np.ndarray,
    min_points: int = 5,
    precomputed_dc_dists=False,
) -> np.ndarray:
    if len(X) != len(labels):
        raise ValueError("Dataset size of `X` differs from label size of `lables`.")

    if len(X) == 0:
        return np.array([])

    if len(X) == 1:
        return np.array([0])

    if len(X) == len(set(labels)):
        return np.zeros(len(X))

    if precomputed_dc_dists:
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("`X` needs to be a distance matrix if `precomputed_dc_dists` is `True`.")
        dc_dists = X
    else:
        dc_dists = DCTree(X, min_points=min_points, no_fastindex=False).dc_distances()

    return silhouette_samples(dc_dists, labels, metric="precomputed")


def p_noise(
    X: np.ndarray,
    labels: np.ndarray,
    min_points: int = 5,
    dc_dists: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (p_sparse, p_far)
    """

    if len(X) == 0:
        raise ValueError("Can't calculate noise score for empty dataset.")
    elif len(X) != len(labels):
        raise ValueError("Dataset size differs from label size.")

    label_set = set(labels)
    # Only noise
    if label_set == {-1}:
        return np.full(len(X), -1), np.full(len(X), -1)
    # No noise
    elif -1 not in label_set:
        return np.array([]), np.array([])

    ## At least one cluster and noise ##
    if dc_dists is None:
        dc_dists = DCTree(X, min_points=min_points, no_fastindex=False).dc_distances()

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
    p_sparse = np.full(len(labels[labels == -1]), np.inf)
    for i in range(len(cluster_ids)):
        numerator = core_dists[labels == -1] - max_core_dist[i]
        denominator = np.maximum(core_dists[labels == -1], max_core_dist[i])
        p_sparse_i = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        p_sparse = np.minimum(p_sparse, p_sparse_i)

    # Distance property of noise evaluation
    p_far = np.full(len(labels[labels == -1]), np.inf)
    for i, id in enumerate(cluster_ids):
        min_dist_to_cluster_i = np.min(dc_dists[np.ix_(labels == -1, labels == id)], axis=1)
        numerator = min_dist_to_cluster_i - max_core_dist[i]
        denominator = np.maximum(min_dist_to_cluster_i, max_core_dist[i])
        p_far_i = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        p_far = np.minimum(p_far, p_far_i)

    # Noise evaluation is minimum of core and distance property
    return p_sparse, p_far
