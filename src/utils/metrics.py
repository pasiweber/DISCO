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
from src.Evaluation.CVDD.cvdd_new import cvdd_score as CVDD
from src.Evaluation.CVNN.cvnn import cvnn_score as CVNN
from src.Evaluation.DSI.dsi import dsi_score as DSI

# Gauss
from sklearn.metrics import silhouette_score as SILHOUETTE
from src.Evaluation.DUNN.dunn import dunn_score as DUNN
from sklearn.metrics import davies_bouldin_score as DB
from sklearn.metrics import calinski_harabasz_score as CH


METRICS = {
    "DISCO": lambda X, l: DISCO(X, l),  ## min_pts
    # "DC_DUNN": DC_DUNN,
    ### Competitors
    "DBCV": lambda X, l: DBCV(X, l),
    "DCSI": lambda X, l: DCSI(X, l),  ## min_pts
    "S_DBW": S_DBW,
    "CDBW": CDBW,
    "CVDD": CVDD,
    "CVNN": CVNN,  ## min_pts
    "DSI": DSI,
    ### Gauss
    "SILHOUETTE": SILHOUETTE,
    "DUNN": DUNN,
    "DB": DB,
    "CH": CH,
}

METRIC_ABBREV = {
    "DISCO": "DISCO",
    # "DC_DUNN": "DC_DUNN",
    ### Competitors
    "DBCV": "DBCV",
    "DCSI": "DCSI",
    "S_DBW": "S_Dbw",
    "CDBW": "CDbw",
    "CVDD": "CVDD",
    "CVNN": "CVNN",
    "DSI": "DSI",
    ### Gauss
    "SILHOUETTE": "SILH.",
    "DUNN": "DUNN",
    "DB": "DB",
    "CH": "CH",
}

METRIC_ABBREV_TABLES = {
    "DISCO": r"DISCO ($\\uparrow$)",
    # "DC_DUNN": r"DC_DUNN ($\\uparrow$)",
    ### Competitors
    "DBCV": r"DBCV ($\\uparrow$)",
    "DCSI": r"DCSI ($\\uparrow$)",
    "S_DBW": r"S_Dbw ($\\downarrow$)",
    "CDBW": r"CDbw ($\\uparrow$)",
    "CVDD": r"CVDD ($\\uparrow$)",
    "CVNN": r"CVNN ($\\downarrow$)",
    "DSI": r"DSI ($\\uparrow$)",
    ### Gauss
    "SILHOUETTE": r"SILH. ($\\uparrow$)",
    "DUNN": r"DUNN ($\\uparrow$)",
    "DB": r"DB ($\\uparrow$)",
    "CH": r"CH ($\\uparrow$)",
}


SELECTED_METRICS = [
    "DISCO",
    # "DC_DUNN",
    ### Competitors
    "DBCV",
    "DCSI",
    # "S_DBW",
    # "CDBW",
    # "CVDD",
    # "CVNN",
    "DSI",
    ### Gauss
    "SILHOUETTE",
    # "DUNN",
    # "DB",
    # "CH",
]
# ["DISCO", "DBCV", "DCSI", "S_DBW", "DSI", "SILHOUETTE", "DUNN"]

RESCALED_METRICS = [
    # "DISCO",
    # "DC_DUNN",
    ### Competitors
    # "DBCV",
    # "DCSI",
    "S_DBW",
    "CDBW",
    "CVDD",
    "CVNN",
    # "DSI",
    ### Gauss
    # "SILHOUETTE",
    # "DUNN",
    # "DB",
    # "CH",
]


def create_and_filter_df(
    eval_results,
    selected_metrics=SELECTED_METRICS,
    excluded_metrics=[],
    sort=False,
):
    df = pd.DataFrame(data=eval_results)
    if selected_metrics:
        df = df[df.measure.isin(selected_metrics)]
    if excluded_metrics:
        df = df[~df.measure.isin(excluded_metrics)]
    if sort:
        df["measure"] = pd.Categorical(df["measure"], selected_metrics)
        df = df.sort_values(["dataset", "measure", "run"])
    return df


def rescale_measures(df, metrics):
    df = df.copy()
    values = df[df.measure.isin(metrics)].groupby(["measure"])["value"]
    df.loc[df.measure.isin(metrics), "value"] = values.transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    return df


def create_and_rescale_df(
    eval_results,
    selected_metrics=SELECTED_METRICS,
    excluded_metrics=[],
    rescale_metrics=RESCALED_METRICS,
    sort=False,
):
    df = create_and_filter_df(
        eval_results,
        selected_metrics=selected_metrics,
        excluded_metrics=excluded_metrics + rescale_metrics,
        sort=sort,
    )
    df_ = create_and_filter_df(
        eval_results,
        selected_metrics=rescale_metrics,
        excluded_metrics=excluded_metrics,
        sort=sort,
    )
    df_ = rescale_measures(df_, rescale_metrics)
    df = pd.concat((df, df_))
    return df
