import numpy as np
import os

from .abstract_datasets import AbstractDatasets, standardize
from clustpy.data import *


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASETS_FOLDER = f"{CURRENT_DIRECTORY}"


### Datasets ###


class Datasets(AbstractDatasets):
    # Tabular data
    Synth_low = "Synth_low"
    Synth_high = "Synth_high"
    HAR = "HAR"
    letterrec = "letterrec."
    htru2 = "htru2"
    Mice = "Mice"
    Pendigits = "Pendigits"
    # Video data
    Weizmann = "Weizmann"
    Keck = "Keck"
    # Image data
    COIL20 = "COIL20"
    COIL100 = "COIL100"
    cmu_faces = "cmu_faces"
    # MNIST data
    Optdigits = "Optdigits"
    USPS = "USPS"
    MNIST = "MNIST"
    FMNIST = "FMNIST"
    KMNIST = "KMNIST"

    @classmethod
    def get_experiments_list(cls):
        return [dataset for dataset in cls if dataset not in cls.__get_excluded()]

    @classmethod
    def __get_excluded(cls):
        return []

    def load_dataset(self):
        match self:
            # Tabular data
            case Datasets.Synth_low:
                path = f"{DATASETS_FOLDER}/low_data_100.npy"
                X, l = np.hsplit(np.load(path), [-1])
                return X, l.reshape(-1)
            case Datasets.Synth_high:
                path = f"{DATASETS_FOLDER}/high_data_100.npy"
                X, l = np.hsplit(np.load(path), [-1])
                return X, l.reshape(-1)
            case Datasets.HAR:
                return load_har(return_X_y=True)
            case Datasets.letterrec:
                return load_letterrecognition(return_X_y=True)
            case Datasets.htru2:
                return load_htru2(return_X_y=True)
            case Datasets.Mice:
                return load_mice_protein(return_X_y=True)
            case Datasets.Pendigits:
                return load_pendigits(return_X_y=True)
            # Video data
            case Datasets.Weizmann:
                X, l = load_video_weizmann(return_X_y=True)
                acts = l[:, 0]
                persons = l[:, 1]
                l = persons * len(np.unique(acts)) + acts
                return X, l
            case Datasets.Keck:
                X, l = load_video_keck_gesture(return_X_y=True, image_size=(100, 100))
                acts = l[:, 0] - 1
                nr_of_acts = len(np.unique(acts)) - 1
                persons = l[:, 1]
                l_new = np.full(len(l), -1)
                l_new[acts != -1] = (persons * nr_of_acts)[acts != -1] + acts[acts != -1]
                return X, l_new
            # Image data
            case Datasets.COIL20:
                return load_coil20(return_X_y=True)
            case Datasets.COIL100:
                return load_coil100(return_X_y=True)
            case Datasets.cmu_faces:
                X, l = load_cmu_faces(return_X_y=True)
                l = l[:, 0]
                return X, l
            # MNIST data
            case Datasets.Optdigits:
                return load_optdigits(return_X_y=True)
            case Datasets.USPS:
                return load_usps(return_X_y=True)
            case Datasets.MNIST:
                return load_mnist(return_X_y=True)
            case Datasets.FMNIST:
                return load_fmnist(return_X_y=True)
            case Datasets.KMNIST:
                return load_kmnist(return_X_y=True)
            case _:
                raise AttributeError

    def standardize_dataset(self, X, l):
        match self:
            case dataset if dataset in [
                # Tabular data
                Datasets.Synth_low,
                Datasets.Synth_high,
                Datasets.HAR,
                Datasets.letterrec,
                Datasets.htru2,
                Datasets.Mice,
                Datasets.Pendigits,
            ]:
                return standardize(X, l, axis=0)
            case dataset if dataset in [
                # Video data
                Datasets.Weizmann,
                Datasets.Keck,
                # Image data
                Datasets.COIL20,
                Datasets.COIL100,
                Datasets.cmu_faces,
                # MNIST data
                Datasets.Optdigits,
                Datasets.USPS,
                Datasets.MNIST,
                Datasets.FMNIST,
                Datasets.KMNIST,
            ]:
                return standardize(X, l, axis=None)
            case _:
                raise AttributeError
