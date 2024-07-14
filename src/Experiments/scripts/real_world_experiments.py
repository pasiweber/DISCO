import numpy as np
import pandas as pd
import os
import sys
import time

from collections import defaultdict
from mpire.pool import WorkerPool
from tqdm import tqdm


DISCO_ROOT_PATH = "/export/share/pascalw777dm/DISCO"
sys.path.append(DISCO_ROOT_PATH)
os.environ["TZ"] = "Europe/Vienna"

from datasets.real_world_datasets import Datasets as RealWorldDatasets
from src.utils.metrics import METRICS as ALL_METRICS
from src.utils.experiments import insert_dict, exec_metric


RESULTS_PATH = f"{DISCO_ROOT_PATH}/results/"
TASK_TIMEOUT = 24 * 60 * 60  # 24 hours
N_JOBS = 10  # 32
RUNS = 1

# del ALL_METRICS["CDBW"]
METRICS = ALL_METRICS


if sys.argv[1] == "normal":
    print("Use data without z-normalization\n")
    config = {
        "save_folder": "real_world",
        "datasets": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
        "id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
        "load_fn_dict": {dataset.name: lambda: dataset.data_cached for dataset in RealWorldDatasets.get_experiments_list()},
        "metrics": METRICS,
    }
elif sys.argv[1] == "standardized":
    print("Use data with z-normalization\n")
    config = {
        "save_folder": "real_world_standardized",
        "datasets": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
        "id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
        "load_fn_dict": {dataset.name: lambda: dataset.standardized_data_cached for dataset in RealWorldDatasets.get_experiments_list()},
        "metrics": METRICS,
    }
else:
    print("Need to select `standardized` or `normal`!\n")
    exit()


def run(save_folder, datasets, id_dict, load_fn_dict, metrics):
    pool = WorkerPool(n_jobs=N_JOBS, use_dill=True)
    async_results = {}

    for dataset in datasets:
        X, l = load_fn_dict[dataset]()
        X = X[l != -1]
        l = l[l != -1]

        np.random.seed(0)
        seeds = np.random.choice(10_000, size=RUNS, replace=False)
        for run in tqdm(range(len(seeds))):
            np.random.seed(seeds[run])
            shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
            X_ = X[shuffle_data_index]
            l_ = l[shuffle_data_index]

            for metric_name, metric_fn in metrics.items():
                path = f"{RESULTS_PATH}{save_folder}/{id_dict[dataset]}/{metric_name}_{run}.csv"
                if os.path.exists(path):
                    print(add_time(f"Skipped - Dataset: {dataset}, Run: {run}, Metric: {metric_name}"))
                    continue
                print(add_time(f"Calc - Dataset: {dataset}, Run: {run}, Metric: {metric_name}"))
                async_idx = (dataset, run, metric_name)
                async_results[async_idx] = pool.apply_async(
                    exec_metric, args=(X_, l_, metric_fn), task_timeout=TASK_TIMEOUT
                )

    while async_results:
        time.sleep(10)
        for async_idx, async_result in list(async_results.items()):
            dataset, run, metric_name = async_idx

            if not async_result.ready():
                print(add_time(f"Waiting for - Dataset: {dataset}, Run: {run}, Metric: {metric_name}"))
                continue

            if not async_result.successful():
                print(add_time(f"Failed - Dataset: {dataset}, Run: {run}, Metric: {metric_name}"))
                del async_results[async_idx]
                continue

            value, real_time, cpu_time = async_result.get()
            del async_results[async_idx]

            print(add_time(f"Finished - Dataset: {dataset}, Run: {run}, Metric: {metric_name}"))
            eval_results = defaultdict(list)
            insert_dict(
                eval_results,
                {
                    "dataset": dataset,
                    "measure": metric_name,
                    "run": run,
                    "value": value,
                    "time": real_time,
                    "process_time": cpu_time,
                },
            )
            path = f"{RESULTS_PATH}{save_folder}/{id_dict[dataset]}/{metric_name}_{run}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df = pd.DataFrame(data=eval_results)
            df.to_csv(path, index=False)

    pool.stop_and_join()
    pool.terminate()

    print()
    print(add_time("Finished."))


def add_time(text):
    return f"{time.strftime("%H:%M:%S")}: {text}"


if __name__ == "__main__":
    time.tzset()
    run(**config)
