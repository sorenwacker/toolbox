import numpy as np
import seaborn as sns
import matplotlib as mpl

from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as pl

from scipy.cluster import hierarchy
from scipy.stats import norm

from pathlib import Path as P


def plot_roc(
    target,
    score,
    cutoff_target=None,
    ax=None,
    pos_label=None,
    add_text=False,
    set_tick_labels=True,
    estimate_random=True,
    with_auc=True,
    **kwargs,
):
    ax = _activate_axis_(ax)
    if cutoff_target is not None:
        target = classify(target, cutoff_target)
    fpr, tpr, _ = roc_curve(target, score, pos_label=pos_label)
    auc = roc_auc_score(target, score)
    pl.plot(fpr, tpr, **kwargs)
    if add_text:
        pl.text(0.75, 0.04, f"AUC={auc:4.2f}", size=8)
    if estimate_random:
        plot_random_roc(target, 200, ax=ax)
    _plot_roc_defaults_(set_tick_labels=set_tick_labels, ax=ax)
    return auc


def _classify(i, value, inverse=False):
    """
    Takes in integer i and a reference value and returns
    True when i <= the reference value and False instead.
    - inverse: True returns
    """
    if inverse is False:
        return i > value
    else:
        return i <= value


classify = np.vectorize(_classify)


def _plot_roc_defaults_(set_tick_labels=True, ax=None, roc_percent=True):
    ax = _activate_axis_(ax)
    if set_tick_labels is False:
        pl.gca().set_yticklabels([])
        pl.gca().set_xticklabels([])
    else:
        if roc_percent:
            pl.xticks([0.2, 0.4, 0.6, 0.8], [20, 40, 60, 80])
            pl.yticks([0.2, 0.4, 0.6, 0.8], [20, 40, 60, 80])
            pl.xlabel("False Positive [%]")
            pl.ylabel("True Positive [%]")
        else:
            pl.xticks([0.2, 0.4, 0.6, 0.8])
            pl.yticks([0.2, 0.4, 0.6, 0.8])
            pl.xlabel("False Positive Rate")
            pl.ylabel("True Positive Rate")

    pl.xlim((0, 1))
    pl.ylim((0, 1))
    plot_diagonal(linestyle="--", color="w")
    return ax


def plot_random_roc(labels, N, ax=None):
    ax = _activate_axis_(ax)
    for i in range(N):
        pl.plot(*_random_roc_(labels), alpha=0.01, linewidth=10, color="k", zorder=0)
    return ax


def plot_diagonal(ax=None, **kwargs):
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    pl.plot(
        [np.min([x0, y0]), np.min([x1, y1])],
        [np.min([x0, y0]), np.min([x1, y1])],
        **kwargs,
    )
    pl.xlim((x0, x1))
    pl.ylim((y0, y1))


def plot_hlines(hlines=None, ax=None, color=None, **kwargs):
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    if hlines is not None:
        if isinstance(hlines, int):
            hlines = [hlines]
        for hline in hlines:
            pl.hlines(hline, x0 - 0.2, x1 + 1.2, color=color, **kwargs)
    pl.xlim((x0, x1))
    pl.ylim((y0, y1))


def plot_vlines(vlines=None, ax=None, color=None, **kwargs):
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    if vlines is not None:
        if not isiterable(vlines):
            vlines = [vlines]
        for vline in vlines:
            pl.vlines(vline, y0 - 0.2, y1 + 1.2, color=color, **kwargs)
    pl.xlim((x0, x1))
    pl.ylim((y0, y1))


def _random_roc_(y_train, ax=None):
    """
    Generates random prediction and returns fpr, tpr used to plot a ROC-curve.
    """
    rand_prob = norm.rvs(size=len(y_train))
    fpr, tpr, _ = roc_curve(y_train, rand_prob, pos_label=1)
    return fpr, tpr


def _activate_axis_(ax=None):
    if ax is not None:
        pl.sca(ax)
    return pl.gca()


def _axis_dimensions_(ax=None):
    ax = _activate_axis_(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x0, x1, y0, y1)


def heatmap(dm, vmin=0, vmax=1):
    """based on heatmap function from
    http://nbviewer.ipython.org/github/herrfz/dataanalysis/
    blob/master/week3/svd_pca.ipynb
    Generates a heatmap from the input matrix.
    """
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist, squareform

    D1 = squareform(pdist(dm, metric="euclidean"))
    D2 = squareform(pdist(dm.T, metric="euclidean"))
    f = pl.figure(figsize=(8, 8))
    # add first dendrogram
    ax1 = f.add_axes([0.09, 0.1, 0.2, 0.6])
    Y = linkage(D1, method="complete")
    Z1 = dendrogram(Y, orientation="left")
    ax1.set_xticks([])
    ax1.set_yticks([])
    # add second dendrogram
    ax2 = f.add_axes([0.3, 0.71, 0.6, 0.2])
    Y = linkage(D2, method="complete")
    Z2 = dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])
    # add matrix plot
    axmatrix = f.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1["leaves"]
    idx2 = Z2["leaves"]
    D = dm[idx1, :]
    D = D[:, idx2]
    axmatrix.matshow(D[::-1], aspect="auto", cmap="hot", vmin=vmin, vmax=vmax)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    # cbar=pl.colorbar(im,shrink=0.77,ticks=np.linspace(vmin,vmax,3))
    return {"ordered": D, "rorder": Z1["leaves"], "corder": Z2["leaves"]}


def legend_outside(ax=None, bbox_to_anchor=None, **kwargs):
    """
    Places the legend outside the current axis.
    ax : matplotlib.axes element
        default: pyplot.gca() is called
    bbox_to_anchor : 2D tuple with numeric values
        default: (1, 1.05)
    """
    if ax is None:
        ax = pl.gca()
    if bbox_to_anchor is None:
        bbox_to_anchor = (1, 1.05)
    ax.legend(bbox_to_anchor=bbox_to_anchor, **kwargs)


def savefig(
    name, notebook_name=None, fmt=["pdf", "png", "svg"], bbox_inches="tight", dpi=300
):
    fig = pl.gcf()
    name = str(name)

    output = P("output")

    if notebook_name:
        prefix = f"{notebook_name}__"
        output = output / notebook_name
    else:
        prefix = ""

    name = prefix + name

    for suffix in fmt:
        _o = output / suffix
        _o.mkdir(parents=True, exist_ok=True)
        suffix = f".{suffix}"
        fn = (_o / name).with_suffix(suffix)
        fig.savefig(fn, bbox_inches=bbox_inches, dpi=dpi)
        print(f"Saved: {fn.resolve()}")


def plot_dendrogram(
    df,
    labels=None,
    orientation="left",
    metric=None,
    color_threshold=0,
    above_threshold_color="k",
    **kwargs,
):

    Z = hierarchy.linkage(df, metric=metric)
    T = hierarchy.to_tree(Z)
    data = hierarchy.dendrogram(
        Z,
        labels=labels,
        orientation=orientation,
        color_threshold=color_threshold,
        above_threshold_color=above_threshold_color,
    )
    ndx = data["leaves"]
    if orientation in ["left", "right"]:
        pl.xticks([])
    if orientation in ["top", "bottom"]:
        pl.xticks([])
    pl.gca().set(frame_on=False)

    return Z, T
