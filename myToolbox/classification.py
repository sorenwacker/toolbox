from matplotlib import pyplot as plt

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