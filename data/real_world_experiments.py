import numpy as np
import pandas as pd
import gc
import os
import sys

from collections import defaultdict

parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)

from data._metrics import METRICS as ALL_METRICS
from data._util_experiments import insert_dict, exec_metric
from data.real_world_datasets import Datasets as RealWorldDatasets

from mpire.pool import WorkerPool
from tqdm import tqdm


if sys.argv[1] == "standardized":
    standardized = True
    print("Use data with z-normalization\n")
elif sys.argv[1] == "normal":
    standardized = False
    print("Use data without normalization\n")
else:
    print("Need to select `standardized` or `normal`!\n")
    exit()


RESULTS_PATH = "./../results/"
TASK_TIMEOUT = 2 * 60 * 60  # 2 hours
N_JOBS = 48
RUNS = 1
METRICS = ALL_METRICS


if standardized:
    SAVE_FOLDER = "real_world_standardized"
    DATASETS = RealWorldDatasets.get_experiments_list()
    LOAD_FN = lambda dataset: dataset.standardized_data
    NAME_FN = lambda dataset: dataset.name
    ID_FN = lambda dataset: dataset.id

else:
    SAVE_FOLDER = "real_world"
    DATASETS = RealWorldDatasets.get_experiments_list()
    LOAD_FN = lambda dataset: dataset.original_data
    NAME_FN = lambda dataset: dataset.name
    ID_FN = lambda dataset: dataset.id


for dataset in DATASETS:
    print("Start", NAME_FN(dataset))

    X, l = LOAD_FN(dataset)

    X = X[l != -1]
    l = l[l != -1]

    np.random.seed(0)
    seeds = np.random.choice(10_000, size=RUNS, replace=False)

    for run in tqdm(range(len(seeds))):
        np.random.seed(seeds[run])
        shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
        X_ = X[shuffle_data_index]
        l_ = l[shuffle_data_index]

        with WorkerPool(n_jobs=N_JOBS, use_dill=True) as pool:

            async_results = {}
            for metric_name, metric_fn in METRICS.items():
                path = f"{RESULTS_PATH}{SAVE_FOLDER}/{ID_FN(dataset)}/{metric_name}_{run}.csv"
                if os.path.exists(path):
                    print(f"Skipped - Dataset: {NAME_FN(dataset)}, Run: {run}, Metric: {metric_name}")
                    continue
                print(f"Calc - Dataset: {NAME_FN(dataset)}, Run: {run}, Metric: {metric_name}")
                async_results[metric_name] = pool.apply_async(
                    exec_metric, args=(X_, l_, metric_fn), task_timeout=TASK_TIMEOUT
                )

            for metric_name in async_results.keys():
                value, real_time, cpu_time = async_results[metric_name].get()
                print(f"Finished - Dataset: {NAME_FN(dataset)}, Run: {run}, Metric: {metric_name}")
                eval_results = defaultdict(list)
                insert_dict(
                    eval_results,
                    {
                        "dataset": NAME_FN(dataset),
                        "measure": metric_name,
                        "run": run,
                        "value": value,
                        "time": real_time,
                        "process_time": cpu_time,
                    },
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                df = pd.DataFrame(data=eval_results)
                df.to_csv(path, index=False)

            gc.collect()

    print("End", NAME_FN(dataset))


print()
print("Finished.")
