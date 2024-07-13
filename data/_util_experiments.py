import numpy as np
import time
import gc
import os
import sys
import pickle

parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)

from collections import defaultdict
from data._metrics import METRICS
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


def exec_metric(X, l, metric_fn, args=[], kwargs={}):
    """Calculate evaluation measures for given metric function and given dataset with data `X` and labels `l`."""

    start_time = time.time()
    start_process_time = time.process_time()
    X, l = np.array(X, dtype=np.float64), np.array(l, dtype=int)
    value = metric_fn(X, l, *args, **kwargs)
    end_process_time = time.process_time()
    end_time = time.time()
    return value, end_time - start_time, end_process_time - start_process_time


def calc_eval_measures(X, l, metrics=METRICS, name=None, task_timeout=None):
    """Calculate all evaluation measures for a given dataset with data `X` and labels `l`."""

    with WorkerPool(n_jobs=48, use_dill=True) as pool:

        async_results = {}

        np.random.seed(0)
        seeds = np.random.choice(10_000, size=10, replace=False)

        for run, seed in enumerate(seeds):
            np.random.seed(seed)
            shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
            X_ = X[shuffle_data_index]
            l_ = l[shuffle_data_index]

            for metric_name, metric_fn in metrics.items():
                async_results[(run, metric_name)] = pool.apply_async(
                    exec_metric, args=(X_, l_, metric_fn), task_timeout=task_timeout
                )

        eval_results = defaultdict(list)

        for run in range(10):
            for metric_name in metrics.keys():
                value, real_time, cpu_time = async_results[(run, metric_name)].get()

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
        gc.collect()
    return eval_results


def calc_eval_measures_for_multiple_datasets(
    data, param_values, metrics=METRICS, task_timeout=None
):
    """Calculates all evaluation measures for all datasets in data.

    Args:
        data: 2d matrix of type [datasets x runs]
    """

    eval_results_all = defaultdict(list)

    for param_value in range(len(param_values)):
        for run in range(len(data[param_value])):
            X, l = data[param_value][run]
            eval_results = calc_eval_measures(
                X, l, metrics=metrics, name=param_values[param_value], task_timeout=task_timeout
            )
            merge_dicts(eval_results_all, eval_results)
    return eval_results_all
