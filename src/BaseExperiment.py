import mpire

import pandas as pd
import time

from mpire import WorkerPool

from src.DataLoader import DataLoader
from src.Evaluation.CDBW.CDBW import CDbw
from src.Evaluation.CVDD.CVDD import CVDDIndex
from src.Evaluation.DCSI.dcsi import dcsiscore
from src.Evaluation.S_Dbw.sdbw import S_Dbw
from src.Evaluation.Silhouette.silhouette import silhouette_score
from src.Evaluation.DBCV.DBCV_Base import validity_index
from src.Evaluation.DISCO.disco import disco


class BaseExperiment(object):

    def __init__(self, exp_name, dataname, min_points, repeat):
        """

        :param exp_name: name of experiment
        :param dataname: name of dataset
        :param min_points: min points parameter for disco
        """
        self.dataloader = DataLoader(dataname)
        self.X = self.dataloader.get_features()
        self.y = self.dataloader.get_labels()
        self.exp_name = exp_name
        self.dataname = dataname
        self.min_points = min_points
        self.repeat = repeat

    def get_X_y(self):
        return self.X, self.y

    def run(self):
        print('Running baseexperiment')
        with mpire.WorkerPool(n_jobs=mpire.cpu_count()) as pool:
        #with mpire.WorkerPool(n_jobs=1) as pool:
            results = pool.imap(self.run_timing, range(0, self.repeat), progress_bar=True)
        results = [res for res in results]
        return results



    def run_timing(self, i):
        X, y = self.get_X_y()
        if i != 0:
            X, y = self.dataloader.get_shuffled()
        st_dbcv = time.process_time()
        dbcv = validity_index(X, y)[0]
        end_dbcv = time.process_time()
        st_disco = time.process_time()
        disco_ = disco(X, y, self.min_points)
        end_disco = time.process_time()
        st_sil = time.process_time()
        silhouette = silhouette_score(X, y)
        end_sil = time.process_time()
        st_dcsi = time.process_time()
        dcsi = dcsiscore(X, y, self.min_points)
        end_dcsi = time.process_time()
        st_cvdd = time.process_time()
        cvdd = CVDDIndex(X,y)
        end_cvdd = time.process_time()
        st_cdbw = time.process_time()
        cdbw = CDbw(X, y)
        end_cdbw = time.process_time()
        st_sdbw = time.process_time()
        sdbw = S_Dbw(X, y)
        end_sdbw = time.process_time()
        results = {'Data': self.dataname,'Min_Points':self.min_points,'Run': i, 'DBCV': dbcv, 'DISCO': disco_,
                   'Silhouette': silhouette, 'DCSI': dcsi, 'CVDD': cvdd, 'CDBW': cdbw, 'SDBW': sdbw,
                   'Time_DBCV': end_dbcv - st_dbcv, 'Time_DISCO': end_disco - st_disco,
                   'Time_Silhouette': end_sil - st_sil, 'Time_DCSI': end_dcsi - st_dcsi, 'Time_CVDD': end_cvdd-st_cvdd,
                   'Time_CDBW': end_cdbw-st_cdbw, 'Time_SDBW': end_sdbw-st_sdbw}
        return results

