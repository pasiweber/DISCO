import numpy as np
import os

from .DENSIRED import datagen
from clustpy.data import *
from enum import Enum


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASETS_FOLDER = f"{CURRENT_DIRECTORY}/datasets"


### Datasets ###

class Datasets(Enum):
    # Synthetic data generated with DENSIRED
    Dataset1 = "Dataset1"
    Dataset2 = "Dataset2"
    DatasetDensiredExample = "DatasetDensiredExample"


    @property
    def id(self):
        return super(Datasets, self).name

    @property
    def name(self):
        return super(Datasets, self).value

    @property
    def data(self):
        """Returns (X, l)"""
        return load_and_cache_dataset(self)

    @property
    def original_data(self):
        """Returns (X, l, skeleton, data)"""
        return load_original_dataset(self)

    @property
    def standardized_data(self):
        """Returns (X, l), with X standardized"""
        X, l, _, _ = load_original_dataset(self)
        return standardize_dataset(self, X, l)


def generate_dataset(dataset_config):
    DIMS = dataset_config.get("DIMS")
    N = dataset_config.get("N")
    CLUSTER_NUMS = dataset_config.get("CLUSTER_NUMS")
    CORE_NUMS = dataset_config.get("CORE_NUMS")
    N_NOISE = dataset_config.get("N_NOISE")
    SEED = dataset_config.get("SEED")

    N_CLUSTERS = len(CLUSTER_NUMS) if CLUSTER_NUMS else None

    kwargs = {
        "dim": DIMS,
        "clunum": N_CLUSTERS,
        "core_num": CORE_NUMS,
        "ratio_noise": N_NOISE / N if N_NOISE else None,
        "clu_ratios": CLUSTER_NUMS,
        "seed": SEED,
    }
    kwargs.update(dataset_config.get("kwargs", {}))

    if kwargs["clu_ratios"] and N_NOISE:
        assert sum(kwargs["clu_ratios"]) + N_NOISE == N, "`CLUSTER_NUMS` + `N_NOISE` needs to sum up to `N`"

    skeleton = datagen.densityDataGen(**kwargs)

    data = skeleton.generate_data(N)
    X = data[:, 0:-1]
    l = data[:, -1]
    return X, l, skeleton, data


def load_original_dataset(dataset_id):
    match dataset_id:
        # Synthetic data generated with DENSIRED

        case Datasets.Dataset1:
            config = {
                "DIMS": 2,
                "N": 550,
                "CLUSTER_NUMS": [250, 250],
                "CORE_NUMS": [50, 50],
                "N_NOISE": 50,
                "SEED": 0,
            }
            return generate_dataset(config)

        case Datasets.Dataset2:
            config = {
                "DIMS": 2,
                "N": 1050,
                "CLUSTER_NUMS": [750, 250],
                "CORE_NUMS": [50, 50],
                "N_NOISE": 50,
                "SEED": 0,
                "kwargs": {
                    "dens_factors": [1, 2],
                },
            }
            return generate_dataset(config)

        case Datasets.DatasetDensiredExample:
            config = {
                "N": 5000,
                "kwargs": {
                    "dim": 2,
                    "ratio_noise": 0.1,
                    "max_retry": 5,
                    "dens_factors": [1, 1, 0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1],
                    "square": True,
                    "clunum": 10,
                    "seed": 6,
                    "core_num": 200,
                    "momentum": [0.5, 0.75, 0.8, 0.3, 0.5, 0.4, 0.2, 0.6, 0.45, 0.7],
                    "branch": [0, 0.05, 0.1, 0, 0, 0.1, 0.02, 0, 0, 0.25],
                    "con_min_dist": 0.8,
                    "verbose": False,
                    "safety": True,
                    "domain_size": 20,
                    "random_start": False,
                },
            }
            return generate_dataset(config)

        case _:
            raise AttributeError


def standardize(X, l, axis=None):
    std = np.std(X, axis=axis)
    X = X[:, std != 0]  # Remove features which are identical over all samples
    mean = np.mean(X, axis=axis)
    X = (X - mean) / std[std != 0]
    return X, l

def standardize_dataset(dataset_id, X, l):
    match dataset_id:
        case dataset if dataset in [
            # Synthetic data generated with DENSIRED
            Datasets.Dataset1,
            Datasets.Dataset2,
            Datasets.DatasetDensiredExample,
        ]:
            return standardize(X, l, axis=0)
        case dataset if dataset in []:
            return standardize(X, l, axis=None)
        case _:
            raise AttributeError


def load_np_dataset(path):
    X = np.load(path + "_data.npy", allow_pickle=True)
    l = np.load(path + "_labels.npy", allow_pickle=True)
    X = X.reshape((len(X), -1))
    return X, l

def load_and_cache_dataset(dataset_id):
    if not os.path.exists(f"{DATASETS_FOLDER}/.cache/"):
        os.makedirs(f"{DATASETS_FOLDER}/.cache/")
    cache_path = f"{DATASETS_FOLDER}/.cache/{dataset_id}"
    if os.path.exists(f"{cache_path}_data.npy") and os.path.exists(f"{cache_path}_labels.npy"):
        return load_np_dataset(cache_path)
    else:
        X, l, _, _ = load_original_dataset(dataset_id)
        np.save(f"{cache_path}_data.npy", X)
        np.save(f"{cache_path}_labels.npy", l)
        return X, l
