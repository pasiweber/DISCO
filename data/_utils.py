import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.neighbors import NearestNeighbors


def insert_dict(dict, key_value_dict):
    for key, value in key_value_dict.items():
        dict[key].append(value)


def exec_metric(metric_fn, X, l):
    start_time = time.time()
    start_process_time = time.process_time()
    value = metric_fn(X, l)
    end_process_time = time.process_time()
    end_time = time.time()
    return value, end_time - start_time, end_process_time - start_process_time


def add_noise(X, l, n_noise, eps):
    noise = np.empty((n_noise, X.shape[1]))
    noise_too_near = np.array(range(len(noise)))
    while len(noise_too_near) > 0:
        noise[noise_too_near] = np.random.uniform(
            np.min(X, axis=0), np.max(X, axis=0), size=(len(noise_too_near), X.shape[1])
        )
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
        dists, _ = nbrs.kneighbors(noise)
        noise_too_near = np.where(dists < eps)[0]

    X_ = np.vstack((X, noise))
    l_ = np.hstack((l, np.array([-1] * len(noise))))

    return X_, l_


def plot_lineplot(
    df,
    x_axis,
    y_axis,
    grouping=None,
    order=None,
    x_range=(None, None),
    y_range=(None, None),
    figsize=(15, 5),
    errorbar="se",
    highlight=1,
):
    plt.figure(figsize=figsize)
    highlight -= 1

    if order is None:
        order = list(df[grouping].unique())

    highlight_index = (
        [highlight] + list(range(0, highlight)) + list(range(highlight + 1, len(order)))
    )
    order = list(np.array(order)[highlight_index])

    def repeat(array):
        return array * ((len(order) - 1 + len(array)) // len(array))

    markers = ["o"] + repeat(["v", "^", "<", ">", "p", "P", "X", "d", "D", "H"])
    palette = ["black"] + repeat(sns.color_palette("bright"))
    sizes = [2] + repeat([1])
    dashes = [(1, 0)] + repeat([(1, 2), (5, 2), (3, 3, 1, 3)])

    ax = sns.lineplot(
        data=df,
        x=x_axis,
        y=y_axis,
        markers=dict(zip(order, markers)),
        # markersize=7,
        hue=grouping,
        palette=dict(zip(order, palette)),
        hue_order=order[::-1],
        style=grouping,
        dashes=dict(zip(order, dashes)),
        size=grouping,
        sizes=dict(zip(order, sizes)),
        errorbar=errorbar,
    )

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    ### Coloring of the plot
    ax.set_facecolor("white")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.grid(color="lightgray")

    ### Legend
    handles, _ = plt.gca().get_legend_handles_labels()
    inverse_index = np.empty(len(order), dtype=int)
    inverse_index[highlight_index] = np.arange(0, len(order))
    leg = plt.legend(
        handles=list(np.array(handles[::-1])[inverse_index]),
        labels=list(np.array(order)[inverse_index]),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        # loc="lower center",
        # bbox_to_anchor=(0.5, 1),
        # fontsize=19,
        # ncol=4,
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")

    plt.tight_layout()


def plot_barplot(
    df,
    x_axis,
    y_axis,
    grouping=None,
    order=None,
    x_range=(None, None),
    y_range=(None, None),
    figsize=(15, 5),
    errorbar="se",
):
    plt.figure(figsize=figsize)#
    sns.set_theme(style="whitegrid", palette="bright")

    ax = sns.barplot(
        df,
        x=x_axis,
        y=y_axis,
        hue=grouping,
        hue_order=order,
        errorbar=errorbar,
    )

    for container in ax.containers:
        tmp_hue = df.loc[df[grouping] == container.get_label()]
        ax.bar_label(container, labels=tmp_hue[y_axis])

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    ### Coloring of the plot
    # ax.spines["bottom"].set_color("black")
    # ax.spines["left"].set_color("black")
    # ax.spines["right"].set_color("white")
    # ax.spines["top"].set_color("white")

    plt.tight_layout()
    sns.reset_orig()
