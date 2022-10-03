import numpy as np

from matplotlib_venn import venn2_circles
from matplotlib_venn import venn2
from matplotlib_venn import venn3_circles
from matplotlib_venn import venn3


def venn_diagram(a, b, c=None, labels=None, colors=None, **kwargs):
    if c is None:
        venn = _venn_diagram2(a, b, set_labels=labels, set_colors=colors, **kwargs)
    else:
        venn = _venn_diagram3(a, b, c, set_labels=labels, set_colors=colors, **kwargs)
    return venn


def _venn_diagram3(a, b, c, **kwargs):

    a = list(set(a))
    b = list(set(b))
    c = list(set(c))

    only_a = len([x for x in a if x not in b + c])
    only_b = len([x for x in b if x not in a + c])
    only_c = len([x for x in c if x not in a + b])

    a_b = len(np.intersect1d(a, b))
    a_c = len(np.intersect1d(a, c))
    b_c = len(np.intersect1d(b, c))

    a_b_c = len([x for x in a if (x in b) and (x in c)])

    venn3(
        subsets=(only_a, only_b, a_b - a_b_c, only_c, a_c - a_b_c, b_c - a_b_c, a_b_c),
        **kwargs
    )
    venn3_circles(
        subsets=(only_a, only_b, a_b - a_b_c, only_c, a_c - a_b_c, b_c - a_b_c, a_b_c),
        linestyle="dashed",
        linewidth=1,
    )


def _venn_diagram2(a, b, **kwargs):

    a = list(set(a))
    b = list(set(b))

    only_a = len([x for x in a if x not in b])
    only_b = len([x for x in b if x not in a])

    a_b = len(np.intersect1d(a, b))

    venn2(subsets=(only_a, only_b, a_b), **kwargs)
    venn2_circles(subsets=(only_a, only_b, a_b), linestyle="dashed", linewidth=1)
