import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors

from src.Evaluation.disco import disco_score as DISCO, p_noise as p_noise
from src.Evaluation.dc_dunn import dc_dunn_score as DC_DUNN

# Competitors
from src.Evaluation.dbcv import validity_index as DBCV
from src.Evaluation.dcsi import dcsi_score as DCSI
from src.Evaluation.s_dbw import sdbw_score as S_DBW
from src.Evaluation.cdbw import cdbw_score as CDBW
from src.Evaluation.cvdd_new import cvdd_score as CVDD
from src.Evaluation.cvnn import cvnn_score as CVNN
from src.Evaluation.dsi import dsi_score as DSI
from src.Evaluation.lccv import lccv_score as LCCV
from src.Evaluation.viasckde import viasckde_score as VIASCKDE

# Gauss
from sklearn.metrics import silhouette_score as SILHOUETTE
from sklearn.metrics import davies_bouldin_score as DB
from sklearn.metrics import calinski_harabasz_score as CH
from src.Evaluation.dunn import dunn_score as DUNN

METRICS = {
    "DISCO": lambda X, l: DISCO(X, l),  ## min_pts
    ### Competitors
    "DBCV": lambda X, l: DBCV(X, l, metric="sqeuclidean"),
}
colors = {
    'blue'       : '#00549F', # blue
    'blue_75'    : '#407FB7',
    'blue_50'    : '#8EBAE5',
    'blue_25'    : '#C7DDF2',
    'blue_10'    : '#E8F1FA',

    'black'      : '#000000', # black
    'black_75'   : '#646567',
    'black_50'   : '#9C9E9F',
    'black_25'   : '#CFD1D2',
    'black_10'   : '#ECEDED',

    'magenta'    : '#E30066', # magenta
    'magenta_75' : '#E96088',
    'magenta_50' : '#F19EB1',
    'magenta_25' : '#F9D2DA',
    'magenta_10' : '#FDEEF0',

    'yellow'     : '#FFED00', # yellow
    'yellow_75'  : '#FFF055',
    'yellow_50'  : '#FFF59B',
    'yellow_25'  : '#FFFAD1',
    'yellow_10'  : '#FFFDEE',

    'petrol'     : '#006165',
    'petrol_75'  : '#2D7F83',
    'petrol_50'  : '#7DA4A7',
    'petrol_25'  : '#BFD0D1',
    'petrol_10'  : '#E6ECEC',

    'turquoise'    : '#0098A1',
    'turquoise_75' : '#00B1B7',
    'turquoise_50' : '#89CCCF',
    'turquoise_25' : '#CAE7E7',
    'turquoise_10' : '#EBF6F6',

    'green'      : '#57AB27', # green
    'green_75'   : '#8DC060',
    'green_50'   : '#B8D698',
    'green_25'   : '#DDEBCE',
    'green_10'   : '#F2F7EC',

    'lime'       : '#BDCD00',
    'lime_75'    : '#D0D95C',
    'lime_50'    : '#E0E69A',
    'lime_25'    : '#F0F3D0',
    'lime_10'    : '#F9FAED',

    'orange'     : '#F6A800',
    'orange_75'  : '#FABE50',
    'orange_50'  : '#FDD48F',
    'orange_25'  : '#FEEAC9',
    'orange_10'  : '#FFF7EA',

    'red'        : '#CC071E',
    'red_75'     : '#D85C41',
    'red_50'     : '#E69679',
    'red_25'     : '#F3CDBB',
    'red_10'     : '#FAEBE3',

    'bordeaux'   : '#A11035',
    'bordeaux_75': '#B65256',
    'bordeaux_50': '#CD8B87',
    'bordeaux_25': '#E5C5C0',
    'bordeaux_10': '#F5E8E5',

    'purple'     : '#612158',
    'purple_75'  : '#834E75',
    'purple_50'  : '#A8859E',
    'purple_25'  : '#D2C0CD',
    'purple_10'  : '#EDE5EA',

    'lila'       : '#7A6FAC', # purple
    'lila_75'    : '#9B91C1',
    'lila_50'    : '#BCB5D7',
    'lila_25'    : '#DEDAEB',
    'lila_10'    : '#F2F0F7',
}



def add_noise_(X, l, n_noise, eps, noise_eps, border=0):
    """Add noise to data with at least eps distance to the data."""

    noise = np.empty((n_noise, X.shape[1]))
    noise_too_near = np.array(range(len(noise)))
    while len(noise_too_near) > 0:
        noise[noise_too_near] = np.random.uniform(-50, 50, size=(len(noise_too_near), X.shape[1]))
        nbrs_points = NearestNeighbors(n_neighbors=1).fit(X)
        dists_points = nbrs_points.kneighbors(noise)[0]
        noise_too_near_points = np.where(dists_points < eps)[0]
        nbrs_noise = NearestNeighbors(n_neighbors=2).fit(noise)
        dists_noise = nbrs_noise.kneighbors(noise)[0][:, 1]
        noise_too_near_noise = np.where(dists_noise < noise_eps)[0]
        noise_too_near = np.unique(np.hstack((noise_too_near_points, noise_too_near_noise)))

    X_ = np.vstack((X, noise))
    l_ = np.hstack((l, np.array([-1] * len(noise))))

    return X_, l_

def plot_data(X, l, save_fig=None, save_format="png", show=True, cluster_marker_size=5, cluster_marker_density=1, noise_marker_size=80):
    fig = plt.figure()
    #cmap = mcolors.ListedColormap([colors["purple"], colors["red"], colors["green"], colors["orange"]])
    cmap = mcolors.ListedColormap(['dimgray','mediumvioletred', 'orange','navy', 'gold'])#colors["blue"]
    plt.scatter(
        X[:, 0][l != -1][::cluster_marker_density],
        X[:, 1][l != -1][::cluster_marker_density],
        s=cluster_marker_size,
        c=l[l != -1][::cluster_marker_density],
        vmin=-1,
        vmax=3,
        cmap=cmap,
    )
    plt.scatter(X[:, 0][l==-1], X[:, 1][l==-1], s=noise_marker_size, c=l[l==-1], vmin=-1, vmax=3, cmap=cmap, marker="+", alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    #if save_fig:
    plt.savefig(f"{save_fig}v2.{save_format}", format=save_format, dpi=300, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    return fig

def print_scores(X, l):
    def get_score(eval_metric):
        try:
            return f"{eval_metric.__name__}: {round(eval_metric(X, l), 2)}"
        except:
            pass

    from src.utils.metrics import METRICS
    for name in METRICS.keys():
        METRICS[name].__name__ = name
    results = []
    for metric in METRICS.values():
        results.append(get_score(metric))
    # results = []
    for result in results:
        print(result)


radii = [15,15]
n_points_per_cluster = [ 300, 300]
centers = np.array([[-40, 20], [40, 20]])
corrected_centers = []
X = np.array(
    [
        [math.sin((2 * math.pi / n_points) * i) * 15+centers[j,0], math.cos((2 * math.pi / n_points) * i) * 15+centers[j,1]]
        for j, n_points in enumerate(n_points_per_cluster)
        for i in range(n_points)
    ]
)
l = np.array(sum([[c] * n_points for c, n_points in enumerate(n_points_per_cluster)], []))

n_noise =400
X, l = add_noise_(X, l, n_noise, 2, 0.5, border=0.5)
l_1 = l.copy()
radius_incl = 10
for i, label in enumerate(l):
    if label ==-1.0:
        value_c1 = np.linalg.norm(X[i]-centers[0])
        value_c2 = np.linalg.norm(X[i]-centers[1])
        # ausserhalb kleiner Radius
        if  value_c1 >= (radii[0] - radius_incl) :
            # innerhalb größerer Radius
            if value_c1 <= (radii[0] + radius_incl):
                l[i]= 0.
        if value_c2 >= (radii[1] - radius_incl) and value_c2 <= (radii[1] + radius_incl):
            l[i]= 1.


fig = plot_data(X, l,save_fig="Motivation_jitter", noise_marker_size=25, show=False, cluster_marker_density=1)
print("Motivation_jitter")
print_scores(X, l)

fig = plot_data(X, l_1,save_fig="Motivation_pureNoise", noise_marker_size=25, show=False, cluster_marker_density=1)
print("Motivation_pureNoise")
print_scores(X, l_1)