import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import euclidean_distances


def dsi_score(data, labels):
    # what kind of distance measure?
    dist = euclidean_distances(data, data)

    dist = np.array(dist)

    # calculate KS similarity between ICD and BCD for every class
    KS_list = []

    for i in np.unique(labels):
        ind_i = np.where(labels == i)[0]

        # ICD set: intra-class distances
        ICD_i = dist[np.ix_(ind_i, ind_i)]
        elements_ICD = np.triu_indices(len(ind_i), k=1)
        ICD_vector = ICD_i[elements_ICD]

        # BCD set: between-class distances
        BCD_vector = dist[np.ix_(ind_i, np.setdiff1d(np.arange(len(labels)), ind_i))].flatten()

        # calculate KS statistic
        KS_i = ks_2samp(ICD_vector, BCD_vector).statistic
        KS_list.append(KS_i)

    # average KS similarity values
    DSI = np.mean(KS_list)

    return DSI