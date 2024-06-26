import mpire

import pandas as pd
import time

from mpire import WorkerPool

from src.DataLoader import DataLoader
from src.Evaluation.DCSI.dcsi import dcsiscore
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

    def run(self, timing=False):
        print('Running baseexperiment')
        if timing:
            with mpire.WorkerPool(n_jobs=mpire.cpu_count()) as pool:
                results = pool.imap(self.run_timing, range(0, self.repeat), progress_bar=True)
        else:
            with mpire.WorkerPool(n_jobs=mpire.cpu_count()) as pool:
                results = pool.imap(self.run_untimed, range(0, self.repeat), progress_bar=True)
        results = [res for res in results]
        return results



    def run_timing(self, i):
        X, y = self.get_X_y()
        if i != 0:
            X, y = self.dataloader.get_shuffled()
        st_dbcv = time.process_time()
        dbcv = validity_index(X, y)
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
        results = {'Data': self.dataname,'Min_Points':self.min_points,'Run': i, 'DBCV': dbcv, 'DISCO': disco_, 'Silhouette': silhouette, 'DCSI': dcsi,
                   'Time_DBCV': end_dbcv - st_dbcv,
                   'Time_DISCO': end_disco - st_disco, 'Time_Silhouette': end_sil - st_sil,
                   'Time_DCSI': end_dcsi - st_dcsi}
        return results

    def run_untimed(self, i):
        X, y = self.get_X_y()
        if i != 0:
            X, y = self.dataloader.get_shuffled()
        #dbcv = validity_index(X, y)
        dbcv = -1
        disco_ = disco(X, y, self.min_points)
        silhouette = silhouette_score(X, y)
        dcsi = dcsiscore(X, y, self.min_points)
        results = {'Data': self.dataname,'Min_Points':self.min_points,'Run': i, 'DBCV': dbcv, 'DISCO': disco_, 'Silhouette': silhouette, 'DCSI': dcsi}
        return results
