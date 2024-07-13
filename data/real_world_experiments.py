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


if sys.argv[1] == "standardized":
    standardized = True
    print("Use data with z-normalization\n")
elif sys.argv[1] == "normal":
    standardized = False
    print("Use data without normalization\n")
else:
    print("Need to select `standardized` or `normal`!\n")
    exit()


for dataset in RealWorldDatasets.get_experiments_list():
    print("Start", dataset.name)

    if standardized:
        X, l = dataset.standardized_data
    else:
        X, l = dataset.original_data

    X = X[l != -1]
    l = l[l != -1]

    np.random.seed(0)
    seeds = np.random.choice(10_000, size=1, replace=False)

    for run in tqdm(range(len(seeds))):
        np.random.seed(seeds[run])
        shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
        X_ = X[shuffle_data_index]
        l_ = l[shuffle_data_index]

        for metric_name, metric_fn in METRICS.items():

            if standardized:
                save_folder = "real_world_standardized"
            else:
                save_folder = "real_world"

            path = f"./../results/{save_folder}/{dataset.id}/{metric_name}_{run}.csv"
            if os.path.exists(path):
                print(f"Skipped - Dataset: {dataset.name}, Run: {run}, Metric: {metric_name}")
                continue

            print(f"Calc - Dataset: {dataset.name}, Run: {run}, Metric: {metric_name}")

            eval_results = defaultdict(list)

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

            os.makedirs(os.path.dirname(path), exist_ok=True)
            df = pd.DataFrame(data=eval_results)
            df.to_csv(path, index=False)

    print("End", dataset.name)


print()
print("Finished.")
