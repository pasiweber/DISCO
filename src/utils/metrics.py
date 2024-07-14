import pandas as pd
import os
import sys

parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)


from src.Evaluation.DISCO.disco import disco_score as DISCO
from src.Evaluation.DISCO.disco import noise_eval as noise_eval
from src.Evaluation.DC_DUNN.dc_dunn import dc_dunn_score as DC_DUNN

# Competitors
from src.Evaluation.DBCV.dbcv import dbcv_score as DBCV
from src.Evaluation.DCSI.dcsi import dcsi_score as DCSI
from src.Evaluation.S_Dbw.sdbw import sdbw_score as S_DBW
from src.Evaluation.CDBW.cdbw import cdbw_score as CDBW
from src.Evaluation.CVDD.cvdd import cvdd_score as CVDD
from src.Evaluation.CVNN.cvnn import cvnn_score as CVNN
from src.Evaluation.DSI.dsi import dsi_score as DSI

# Gauss
from sklearn.metrics import silhouette_score as SILHOUETTE
from src.Evaluation.DUNN.dunn import dunn_score as DUNN
from sklearn.metrics import davies_bouldin_score as DB
from sklearn.metrics import calinski_harabasz_score as CH


METRICS = {
    "DISCO": lambda X, l: DISCO(X, l),  ## min_pts
    "DC_DUNN": DC_DUNN,
    # Competitors
    "DBCV": lambda X, l: DBCV(X, l),
    "DCSI": lambda X, l: DCSI(X, l),  ## min_pts
    "S_DBW": S_DBW,
    "CDBW": CDBW,
    "CVDD": CVDD,
    "CVNN": CVNN,  ## min_pts
    "DSI": DSI,
    # Gauss
    "SILHOUETTE": SILHOUETTE,
    "DUNN": DUNN,
    "DB": DB,
    "CH": CH,
}


SELECTED_METRICS = [
    "DISCO",
    # "DC_DUNN",
    ### Competitors
    "DBCV",
    "DCSI",
    # "S_DBW",
    "CDBW",
    "CVDD",
    # "CVNN",
    "DSI",
    ### Gauss
    "SILHOUETTE",
    # "DUNN",
    # "DB",
    # "CH",
]


def create_and_filter_df(eval_results, selected_metrics=SELECTED_METRICS, excluded_metrics=None, sort=False):
    df = pd.DataFrame(data=eval_results)
    if selected_metrics:
        df = df[df.measure.isin(selected_metrics)]
    if excluded_metrics:
        df = df[~df.measure.isin(excluded_metrics)]
    if sort:
        df["measure"] = pd.Categorical(df["measure"], selected_metrics)
        df = df.sort_values(["dataset", "measure", "run"])
    return df
