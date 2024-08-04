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

from src.utils.experiments import insert_dict, exec_func


def exec_clustering_(shared_objects, dataset_name, run, func_name, task_timeout):
    datasets, functions = shared_objects
    try:
        return exec_func(datasets[(dataset_name, run)], functions[func_name])
    except TimeoutError as e:
        print(add_time(f"Timeout - Dataset: {dataset_name}, Run: {run}, Metric: {func_name} - `{e}`"))
        return np.nan, task_timeout, task_timeout
    except Exception as e:
        print(add_time(f"Error - Dataset: {dataset_name}, Run: {run}, Metric: {func_name} - `{e}`"))
        return np.nan, np.nan, np.nan

def run_multiple_experiments(
    save_folder,
    dataset_names,
    dataset_id_dict,
    dataset_load_fn_dict,
    functions,
    runs=10,
    n_jobs=-1,
    task_timeout=12 * 60 * 60,  # 12 hours
):
    datasets = {}
    for dataset_name in dataset_names:
        X, l = dataset_load_fn_dict[dataset_name]()
        np.random.seed(0)
        seeds = np.random.choice(10_000, size=runs, replace=False)
        for run in range(runs):
            np.random.seed(seeds[run])
            shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
            X_ = X[shuffle_data_index]
            l_ = l[shuffle_data_index]
            datasets[(dataset_name, run)] = (X_, l_)

    async_results = {}
    pool = WorkerPool(n_jobs=n_jobs, use_dill=True, shared_objects=(datasets, functions))
    for dataset_name in dataset_names:
        for run in range(runs):
            for func_name in functions:
                path = f"{save_folder}/{dataset_id_dict[dataset_name]}/{func_name}_{run}.csv"
                if os.path.exists(path):
                    print(add_time(f"Skipped - Dataset: {dataset_name}, Run: {run}, Metric: {func_name}"))
                    continue
                print(add_time(f"Calc - Dataset: {dataset_name}, Run: {run}, Metric: {func_name}"))
                async_idx = (dataset_name, run, func_name)
                async_results[async_idx] = pool.apply_async(
                    exec_clustering_, args=(dataset_name, run, func_name, task_timeout), task_timeout=task_timeout
                )

    while async_results:
        print(add_time("-----"))
        time.sleep(10)
        current_tasks = 0
        for async_idx, async_result in list(async_results.items()):
            dataset_name, run, func_name = async_idx

            if not async_result.ready():
                current_tasks += 1
                if current_tasks <= n_jobs:
                    print(add_time(f"Calculating - Dataset: {dataset_name}, Run: {run}, Metric: {func_name}"))
                continue

            if not async_result.successful():
                print(add_time(f"Failed - Dataset: {dataset_name}, Run: {run}, Metric: {func_name}"))
                del async_results[async_idx]
                continue

            value, real_time, cpu_time = async_result.get()
            del async_results[async_idx]

            print(add_time(f"Finished - Dataset: {dataset_name}, Run: {run}, Metric: {func_name} - `{value}`"))
            eval_results = defaultdict(list)
            insert_dict(
                eval_results,
                {
                    "dataset": dataset_name,
                    "measure": func_name,
                    "run": run,
                    "value": value,
                    "time": real_time,
                    "process_time": cpu_time,
                },
            )
            path = f"{save_folder}/{dataset_id_dict[dataset_name]}/{func_name}_{run}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df = pd.DataFrame(data=eval_results)
            df.to_csv(path, index=False, na_rep='nan')

    print(add_time("-----"))
    pool.stop_and_join()
    pool.terminate()

    print()
    print(add_time("Finished."))


def add_time(text):
    return f"{time.strftime("%H:%M:%S")}: {text}"
