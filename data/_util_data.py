import numpy as np
from sklearn.neighbors import NearestNeighbors


def add_noise(X, l, n_noise, eps):
    """Add noise to data with at least eps distance to the data."""

    noise = np.empty((n_noise, X.shape[1]))
    noise_too_near = np.array(range(len(noise)))
    while len(noise_too_near) > 0:
        noise[noise_too_near] = np.random.uniform(
            np.min(X, axis=0), np.max(X, axis=0), size=(len(noise_too_near), X.shape[1])
        )
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
        dists, _ = nbrs.kneighbors(noise)
        noise_too_near = np.where(dists < eps)[0]

    X_ = np.vstack((X, noise))
    l_ = np.hstack((l, np.array([-1] * len(noise))))

    return X_, l_
