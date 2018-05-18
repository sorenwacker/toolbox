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
    Z1 = dendrogram(Y, orientation='right')
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


def _activate_axis_(ax=None):
    if ax is not None:
        plt.sca(ax)
    return plt.gca()



def _axis_dimensions_(ax=None):
    ax = _activate_axis_(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x0, x1, y0, y1) 