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
from datasets.density_datasets import Datasets as DensityDatasets
from src.utils.metrics import METRICS as ALL_METRICS
from src.utils.experiments import insert_dict, exec_metric


RESULTS_PATH = f"{DISCO_ROOT_PATH}/results/"
TASK_TIMEOUT = 12 * 60 * 60  # 6 hours

# del ALL_METRICS["CDBW"]
# del ALL_METRICS["CVDD"]
# del ALL_METRICS["LCCV"]
# del ALL_METRICS["VIASCKDE"]

METRICS = ALL_METRICS.copy()


if sys.argv[1] == "normal":
    print("Use data without z-normalization\n")
    config = {
        "save_folder": "real_world",
        "dataset_names": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.data_cached for dataset in RealWorldDatasets.get_experiments_list()},
        "metrics": METRICS,
    }
    N_JOBS = 1
    RUNS = 1
    del ALL_METRICS["CDBW"]

elif sys.argv[1] == "standardized":
    print("Use data with z-normalization\n")
    config = {
        "save_folder": "real_world_standardized",
        "dataset_names": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.standardized_data_cached for dataset in RealWorldDatasets.get_experiments_list()},
        "metrics": METRICS,
    }
    N_JOBS = 1
    RUNS = 1
    del ALL_METRICS["CDBW"]


elif sys.argv[1] == "density_normal":
    print("Use data without z-normalization\n")
    config = {
        "save_folder": "density",
        "dataset_names": [dataset.name for dataset in DensityDatasets],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in DensityDatasets},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.data_cached for dataset in DensityDatasets},
        "metrics": METRICS,
    }
    N_JOBS = 32
    RUNS = 10

elif sys.argv[1] == "density_standardized":
    print("Use data with z-normalization\n")
    config = {
        "save_folder": "density_standardized",
        "dataset_names": [dataset.name for dataset in DensityDatasets],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in DensityDatasets},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.standardized_data_cached for dataset in DensityDatasets},
        "metrics": METRICS,
    }
    N_JOBS = 32
    RUNS = 10

else:
    print("Need to select `standardized` or `normal`!\n")
    exit()



def exec_metric_(shared_objects, dataset_name, run, metric_name):
    datasets, metrics = shared_objects
    try:
        return exec_metric(datasets[(dataset_name, run)], metrics[metric_name])
    except TimeoutError as e:
        print(add_time(f"Timeout - Dataset: {dataset_name}, Run: {run}, Metric: {metric_name} - {e}"))
    except Exception as e:
        print(add_time(f"Error - Dataset: {dataset_name}, Run: {run}, Metric: {metric_name} - `{e}`"))
    return np.nan, np.nan, np.nan

def run(save_folder, dataset_names, dataset_id_dict, dataset_load_fn_dict, metrics):
    datasets = {}
    for dataset_name in dataset_names:
        X, l = dataset_load_fn_dict[dataset_name]()
        X = X[l != -1]
        l = l[l != -1]
        np.random.seed(0)
        seeds = np.random.choice(10_000, size=RUNS, replace=False)
        for run in range(RUNS):
            np.random.seed(seeds[run])
            shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
            X_ = X[shuffle_data_index]
            l_ = l[shuffle_data_index]
            datasets[(dataset_name, run)] = (X_, l_)

    async_results = {}
    pool = WorkerPool(n_jobs=N_JOBS, use_dill=True, shared_objects=(datasets, metrics))
    for dataset_name in dataset_names:
        for run in range(RUNS):
            for metric_name in metrics:
                path = f"{RESULTS_PATH}{save_folder}/{dataset_id_dict[dataset_name]}/{metric_name}_{run}.csv"
                if os.path.exists(path):
                    print(add_time(f"Skipped - Dataset: {dataset_name}, Run: {run}, Metric: {metric_name}"))
                    continue
                print(add_time(f"Calc - Dataset: {dataset_name}, Run: {run}, Metric: {metric_name}"))
                async_idx = (dataset_name, run, metric_name)
                async_results[async_idx] = pool.apply_async(
                    exec_metric_, args=(dataset_name, run, metric_name), task_timeout=TASK_TIMEOUT
                )

    while async_results:
        print(add_time("-----"))
        time.sleep(10)
        current_tasks = 0
        for async_idx, async_result in list(async_results.items()):
            dataset_name, run, metric_name = async_idx

            if not async_result.ready():
                current_tasks += 1
                if current_tasks <= N_JOBS:
                    print(add_time(f"Calculating - Dataset: {dataset_name}, Run: {run}, Metric: {metric_name}"))
                continue

            if not async_result.successful():
                print(add_time(f"Failed - Dataset: {dataset_name}, Run: {run}, Metric: {metric_name}"))
                del async_results[async_idx]
                continue

            value, real_time, cpu_time = async_result.get()
            del async_results[async_idx]

            print(add_time(f"Finished - Dataset: {dataset_name}, Run: {run}, Metric: {metric_name}"))
            eval_results = defaultdict(list)
            insert_dict(
                eval_results,
                {
                    "dataset": dataset_name,
                    "measure": metric_name,
                    "run": run,
                    "value": value,
                    "time": real_time,
                    "process_time": cpu_time,
                },
            )
            path = f"{RESULTS_PATH}{save_folder}/{dataset_id_dict[dataset_name]}/{metric_name}_{run}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df = pd.DataFrame(data=eval_results)
            if value != np.nan:
                df.to_csv(path, index=False)

    print(add_time("-----"))
    pool.stop_and_join()
    pool.terminate()

    print()
    print(add_time("Finished."))


def add_time(text):
    return f"{time.strftime("%H:%M:%S")}: {text}"


if __name__ == "__main__":
    time.tzset()
    run(**config)
