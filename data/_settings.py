from src.Evaluation.CDBW.cdbw import cdbw_score as CDBW
from src.Evaluation.CVDD.cvdd import cvdd_score as CVDD
from src.Evaluation.CVNN.cvnn import cvnn_score as CVNN
from src.Evaluation.DBCV.dbcv import dbcv_score as DBCV
from src.Evaluation.DCSI.dcsi import dcsi_score as DCSI
from src.Evaluation.DISCO.disco import disco_score as DISCO
from src.Evaluation.DC_DUNN.dc_dunn import dc_dunn_score as DC_DUNN
from src.Evaluation.DUNN.dunn import dunn_score as DUNN
from src.Evaluation.DSI.dsi import dsi_score as DSI
from src.Evaluation.S_Dbw.sdbw import sdbw_score as S_DBW

from sklearn.metrics import silhouette_score as SILHOUETTE
from sklearn.metrics import davies_bouldin_score as DB_SKLEARN
from sklearn.metrics import calinski_harabasz_score as CH_SKLEARN


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
    "DB": DB_SKLEARN,
    "CH": CH_SKLEARN,
}
