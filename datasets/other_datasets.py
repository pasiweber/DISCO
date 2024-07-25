import numpy as np
import os
from .abstract_datasets import AbstractDatasets, standardize
import pickle


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASETS_FOLDER = f"{CURRENT_DIRECTORY}"


### Datasets ###


class Datasets(AbstractDatasets):
    graphdino = "graphdino"

    def load_dataset(self):
        match self:
            case self.graphdino:
                with open("graphdino_morphological_embeddings.pkl", "rb") as f:
                    data = pickle.load(f)
                return data["latent_emb"], np.array(data["split_index"])
            case _:
                raise AttributeError

    def standardize_dataset(self, X, l):
        return standardize(X, l, axis=0)
