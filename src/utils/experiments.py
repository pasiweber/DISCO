import numpy as np
import time
import os
import sys
import pickle

parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)

from collections import defaultdict
from src.utils.metrics import METRICS
from mpire.pool import WorkerPool


CACHE_FOLDER = "./.cache/"


def cache(filename, func, args=[], kwargs={}, recalc=False):
    cache_folder = os.path.abspath(CACHE_FOLDER)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    cache_path = f"{cache_folder}/{filename}.pkl"
    if not recalc and os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    else:
        result = func(*args, **kwargs)
        with open(cache_path, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return result


def insert_dict(dict, key_value_dict):
    for key, value in key_value_dict.items():
        dict[key].append(value)


def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        dict1[key] += value


def exec_metric(data, metric_fn, args=[], kwargs={}):
    """Calculate evaluation measures for given metric function and given dataset with data `X` and labels `l`."""

    start_time = time.time()
    start_process_time = time.process_time()
    value = metric_fn(*data, *args, **kwargs)
    end_process_time = time.process_time()
    end_time = time.time()
    return value, end_time - start_time, end_process_time - start_process_time


def calc_eval_measures(X, l, name=None, metrics=METRICS, runs=10, n_jobs=32, task_timeout=None):
    """Calculate all evaluation measures for a given dataset with data `X` and labels `l`."""

    pool = WorkerPool(n_jobs=n_jobs, use_dill=True)
    async_results = {}

    np.random.seed(0)
    seeds = np.random.choice(10_000, size=runs, replace=False)

    for run, seed in enumerate(seeds):
        np.random.seed(seed)
        shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
        X_ = X[shuffle_data_index]
        l_ = l[shuffle_data_index]

        for metric_name, metric_fn in metrics.items():
            async_idx = (run, metric_name)
            async_results[async_idx] = pool.apply_async(
                exec_metric, args=((X_, l_), metric_fn), task_timeout=task_timeout
            )

    eval_results = defaultdict(list)
    for async_idx, async_result in async_results.items():
        (run, metric_name) = async_idx
        value, real_time, cpu_time = async_result.get()
        insert_dict(
            eval_results,
            {
                "dataset": name,
                "measure": metric_name,
                "run": run,
                "value": value,
                "time": real_time,
                "process_time": cpu_time,
            },
        )

    pool.stop_and_join()
    pool.terminate()
    return eval_results


def calc_eval_measures_for_multiple_datasets(
    data, param_values, metrics=METRICS, n_jobs=32, task_timeout=None
):
    """Calculates all evaluation measures for all datasets in data.

    Args:
        data: 2d matrix of type [datasets x runs]
    """

    pool = WorkerPool(n_jobs=n_jobs, use_dill=True)
    async_results = {}

    for param_value in range(len(param_values)):
        for run in range(len(data[param_value])):
            X, l = data[param_value][run]

            for metric_name, metric_fn in metrics.items():
                async_idx = (param_value, run, metric_name)
                async_results[async_idx] = pool.apply_async(
                    exec_metric, args=((X, l), metric_fn), task_timeout=task_timeout
                )

    eval_results = defaultdict(list)
    for async_idx, async_result in async_results.items():
        (param_value, run, metric_name) = async_idx
        value, real_time, cpu_time = async_result.get()
        insert_dict(
            eval_results,
            {
                "dataset": param_values[param_value],
                "measure": metric_name,
                "run": run,
                "value": value,
                "time": real_time,
                "process_time": cpu_time,
            },
        )

    pool.stop_and_join()
    pool.terminate()
    return eval_results
