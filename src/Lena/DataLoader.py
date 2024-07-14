import json

import pandas as pd
import numpy as np


class DataLoader(object):

    def __init__(self, dataname):
        """
            Construct DataLoader object and set corresponding to dataname name provided.

            Parameters:
                dataname (str): name of the dataset.

        """
        self.__name = dataname
        with open('config/data/{}.json'.format(dataname)) as json_file:
            self.__data_config = json.load(json_file)
        self.__data = pd.read_csv(self.__data_config['file_name'])
        self.__labels = self.__data[self.__data_config['target']].to_numpy()
        self.__features = self.__data.loc[:, self.__data.columns != self.__data_config['target']].to_numpy()

    def get_data(self):
        return self.__data

    def get_labels(self):
        return self.__labels

    def get_features(self):
        return self.__features

    def get_shuffled(self):
        shuffled_data = self.__data.copy()
        shuffled_data = shuffled_data.sample(frac=1)
        # we need labels
        shuffled_labels = shuffled_data[self.__data_config['target']].to_numpy()
        shuffled_features = shuffled_data.loc[:, shuffled_data.columns != self.__data_config['target']].to_numpy()
        return shuffled_features, shuffled_labels
