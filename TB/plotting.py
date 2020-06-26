import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


def plot_roc(target, score, cutoff_target=None, ax=None, pos_label=None, 
             set_tick_labels=True, estimate_random=True, **kwargs):
    ax = _activate_axis_(ax)
    if cutoff_target is not None:
        target = classify(target, cutoff_target)
    fpr, tpr, _ = roc_curve(target, score, pos_label=pos_label)
    plt.plot(fpr, tpr, **kwargs)
    if estimate_random:
        plot_random_roc(target, 200, ax=ax)
    _plot_roc_defaults_(set_tick_labels=set_tick_labels, ax=ax)
    return ax

def _plot_roc_defaults_(set_tick_labels=True, ax=None):
    ax = _activate_axis_(ax)
    if set_tick_labels is False:
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
    else:
        plt.xticks([0.2, 0.4, 0.6, 0.8])
        plt.yticks([0.2, 0.4, 0.6, 0.8]) 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')        
    plt.xlim((0,1))
    plt.ylim((0,1))
    plot_diagonal(linestyle='--', color='w')     
    return ax

def plot_random_roc(labels, N, ax=None):
    ax = _activate_axis_(ax)
    for i in range(N):
        plt.plot(*_random_roc_(labels), alpha=0.01, linewidth=10, color='k', zorder=0)
    return ax

def plot_diagonal(ax=None, **kwargs):
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    plt.plot([np.min([x0, y0]), np.min([x1, y1])],
             [np.min([x0, y0]), np.min([x1, y1])], **kwargs)
    plt.xlim((x0, x1))
    plt.ylim((y0, y1))

def plot_hlines(hlines=None, ax=None, color=None, **kwargs):
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    if hlines is not None:
        if isinstance(hlines, int):
            hlines = [hlines]     
        for hline in hlines:
            plt.hlines(hline, x0-0.2, x1+1.2, color=color, **kwargs)
    plt.xlim((x0, x1))
    plt.ylim((y0, y1)) 

def plot_vlines(vlines=None, ax=None, color=None, **kwargs):
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    if vlines is not None:
        if not isiterable(vlines):
            vlines = [vlines]
        for vline in vlines:
            plt.vlines(vline, y0-0.2, y1+1.2, color=color, **kwargs)
    plt.xlim((x0, x1))
    plt.ylim((y0, y1))

def _random_roc_(y_train, ax=None):
    '''
    Generates random prediction and returns fpr, tpr used to plot a ROC-curve.
    '''
    rand_prob = norm.rvs(size=len(y_train))
    fpr, tpr, _ = roc_curve(y_train, rand_prob, pos_label=1)
    return fpr, tpr

def _activate_axis_(ax=None):
    if ax is not None:
        plt.sca(ax)
    return plt.gca()

def _axis_dimensions_(ax=None):
    ax = _activate_axis_(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x0, x1, y0, y1)

def heatmap(dm, vmin=0, vmax=1):
    '''based on heatmap function from
    http://nbviewer.ipython.org/github/herrfz/dataanalysis/
    blob/master/week3/svd_pca.ipynb
    Generates a heatmap from the input matrix.
    '''
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist, squareform
    D1 = squareform(pdist(dm, metric='euclidean'))
    D2 = squareform(pdist(dm.T, metric='euclidean'))
    f = plt.figure(figsize=(8, 8))
    # add first dendrogram
    ax1 = f.add_axes([0.09, 0.1, 0.2, 0.6])
    Y = linkage(D1, method='complete')
    Z1 = dendrogram(Y, orientation='left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # add second dendrogram
    ax2 = f.add_axes([0.3, 0.71, 0.6, 0.2])
    Y = linkage(D2, method='complete')
    Z2 = dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])
    # add matrix plot
    axmatrix = f.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = dm[idx1, :]
    D = D[:, idx2]
    axmatrix.matshow(D[::-1], aspect='auto', cmap='hot',
                     vmin=vmin, vmax=vmax)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    # cbar=plt.colorbar(im,shrink=0.77,ticks=np.linspace(vmin,vmax,3))
    return {'ordered': D, 'rorder': Z1['leaves'], 'corder': Z2['leaves']}

def legend_outside(ax=None, bbox_to_anchor=None, **kwargs):
    """
    Places the legend outside the current axis.
    ax : matplotlib.axes element
        default: pyplot.gca() is called
    bbox_to_anchor : 2D tuple with numeric values
        default: (1, 1.05)
    """
    if ax is None:
        ax = plt.gca()
    if bbox_to_anchor is None:
        bbox_to_anchor=(1, 1.05)
    ax.legend(bbox_to_anchor=bbox_to_anchor, **kwargs)
    


def hierachical_clustering(df, plot=True, vmin=None, vmax=None,
                           metric='euclidean'):
    '''based on heatmap function from
    http://nbviewer.ipython.org/github/herrfz/dataanalysis/
    blob/master/week3/svd_pca.ipynb
    Generates a heatmap from the input matrix.
    '''
    
    no_plot = not plot
    

    D1 = squareform(pdist(df, metric=metric))
    D2 = squareform(pdist(df.T, metric=metric))
    
    if plot:
        f = plt.figure(figsize=(8, 8), dpi=600)
        # add first dendrogram
        ax1 = f.add_axes([0.61, 0.1, 0.2, 0.6])
    Y = linkage(D1, method='complete')
    Z1 = dendrogram(Y, orientation='right', no_plot=no_plot)
    
    if plot:
        ax1.set_xticks([])
        ax1.set_yticks([])
        # add second dendrogram
        ax2 = f.add_axes([0.0, 0.71, 0.6, 0.2])
        
    Y = linkage(D2, method='complete')
    Z2 = dendrogram(Y, no_plot=no_plot)
    if plot:
        ax2.set_xticks([])
        ax2.set_yticks([])
        # add matrix plot
        axmatrix = f.add_axes([0.0, 0.1, 0.6, 0.6])
        
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = df.iloc[idx1, idx2]
    if plot:
        sns.heatmap(D[::-1], cbar=False)
    return {'ordered': D, 'rorder': Z1['leaves'], 'corder': Z2['leaves']}
