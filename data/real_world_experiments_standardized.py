import numpy as np
import pandas as pd
import os
import sys

from collections import defaultdict

parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)

from data._metrics import METRICS
from data._utils import insert_dict, exec_metric
from data.real_world_datasets import Datasets as RealWorldDatasets

from tqdm import tqdm


for dataset in RealWorldDatasets:
    if os.path.exists(f"./../results/real_world_standardized/{dataset.id}.csv"):
        print("Skipped", dataset.name)
        continue

    print("Start", dataset.name)
    X, l = dataset.data

    X = X[l != -1]
    l = l[l != -1]

    eval_results = defaultdict(list)

    np.random.seed(0)
    seeds = np.random.choice(10_000, size=10, replace=False)

    for run in tqdm(range(len(seeds))):
        np.random.seed(seeds[run])
        shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
        X_ = X[shuffle_data_index]
        l_ = l[shuffle_data_index]

        for metric_name, metric_fn in METRICS.items():
            (value, real_time, cpu_time) = exec_metric(metric_fn, X_, l_)
            insert_dict(
                eval_results,
                {
                    "dataset": dataset.name,
                    "measure": metric_name,
                    "run": run,
                    "value": value,
                    "time": real_time,
                    "process_time": cpu_time,
                },
            )

    df = pd.DataFrame(data=eval_results)
    df.to_csv(f"./../results/real_world_standardized/{dataset.id}.csv", index=False)

    print("End", dataset.name)


print()
print("Finished.")
