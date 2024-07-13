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

from mpire.pool import WorkerPool


with WorkerPool(n_jobs=10, use_dill=True) as pool:

    async_results = {}

    for dataset in RealWorldDatasets:
        if os.path.exists(f"./../results/real_world/{dataset.id}.csv"):
            print("Skipped", dataset.name)
            continue

        print("Start", dataset.name)
        X, l = dataset.data

        X = X[l != -1]
        l = l[l != -1]

        np.random.seed(0)
        seeds = np.random.choice(10_000, size=10, replace=False)

        for run, seed in enumerate(seeds):
            np.random.seed(seed)
            shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
            X_ = X[shuffle_data_index]
            l_ = l[shuffle_data_index]

            for metric_name, metric_fn in METRICS.items():
                async_results[(dataset, run, metric_name)] = pool.apply_async(
                    exec_metric, args=(X_, l_, metric_fn)
                )

    for dataset in RealWorldDatasets:
        if os.path.exists(f"./../results/real_world/{dataset.id}.csv"):
            continue

        print("Gather", dataset.name)

        eval_results = defaultdict(list)

        for run in range(10):
            for metric_name in METRICS.keys():
                value, real_time, cpu_time = async_results[(dataset, run, metric_name)].get()

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
        df.to_csv(f"./../results/real_world/{dataset.id}.csv", index=False)

        print("End", dataset.name)


print()
print("Finished.")
