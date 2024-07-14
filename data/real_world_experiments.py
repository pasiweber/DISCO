import numpy as np
import pandas as pd
import os
import sys
import time

os.environ["TZ"] = "Europe/Vienna"
time.tzset()

def add_time(text):
    return f"{time.strftime("%H:%M:%S")}: {text}"

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
    print("Use data without z-normalization\n")
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


pool = WorkerPool(n_jobs=N_JOBS, use_dill=True)


for dataset in DATASETS:
    print(add_time(f"Start {NAME_FN(dataset)}"))

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

        async_results = {}
        for metric_name, metric_fn in METRICS.items():
            path = f"{RESULTS_PATH}{SAVE_FOLDER}/{ID_FN(dataset)}/{metric_name}_{run}.csv"
            if os.path.exists(path):
                print(add_time(f"Skipped - Dataset: {NAME_FN(dataset)}, Run: {run}, Metric: {metric_name}"))
                continue
            print(add_time(f"Calc - Dataset: {NAME_FN(dataset)}, Run: {run}, Metric: {metric_name}"))
            async_results[metric_name] = pool.apply_async(
                exec_metric, args=(X_, l_, metric_fn), task_timeout=TASK_TIMEOUT
            )

        while async_results:
            time.sleep(10)
            for metric_name in list(async_results.keys()):

                if not async_results[metric_name].ready():
                    print(add_time(f"Waiting for - Dataset: {NAME_FN(dataset)}, Run: {run}, Metric: {metric_name}"))
                    continue

                if not async_results[metric_name].successful():
                    print(add_time(f"Failed - Dataset: {NAME_FN(dataset)}, Run: {run}, Metric: {metric_name}"))
                    del async_results[metric_name]
                    continue

                value, real_time, cpu_time = async_results[metric_name].get()
                del async_results[metric_name]

                print(add_time(f"Finished - Dataset: {NAME_FN(dataset)}, Run: {run}, Metric: {metric_name}"))
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
                path = f"{RESULTS_PATH}{SAVE_FOLDER}/{ID_FN(dataset)}/{metric_name}_{run}.csv"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                df = pd.DataFrame(data=eval_results)
                df.to_csv(path, index=False)

    print(add_time(f"End {NAME_FN(dataset)}"))

pool.stop_and_join()
pool.terminate()

print()
print(add_time("Finished."))
