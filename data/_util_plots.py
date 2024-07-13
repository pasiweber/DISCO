import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def plot_datasets(data, param_values, rows=5, cols=5, figsize=2.0):
    """Plots all datasets in data with corresponding param_value as title.
    `fig_x` columns and `fig_y` rows.

    Args:
        data: 2d matrix of type [datasets x runs]
        param_values: 1d matrix with parameter values per dataset. Used for title.
    """

    fig = plt.figure(
        figsize=(figsize * cols, (figsize + 0.2) * rows),
        layout="tight",
    )
    G = gridspec.GridSpec(rows, cols)

    length = min(len(data), len(param_values), cols * rows)
    data = data[:length]
    param_values = param_values[:length]

    for param_value in range(0, len(data)):
        ax = plt.subplot(G[param_value // cols, param_value % cols])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{param_values[param_value]}")
        X, l = data[param_value][0]
        ax.scatter(X[:, 0], X[:, 1], s=1, c=l)

    return fig


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
    """Plot a line plot for a dataframe."""

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
    """Plot a barplot for a dataframe."""

    plt.figure(figsize=figsize)
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
