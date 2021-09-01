import numpy as np
from matplotlib_venn import venn3, venn2


def venn_diagram(a, b, c=None, labels=None):
    if c is None:
        return _venn_diagram2(a, b, set_labels=labels)
    else:
        return _venn_diagram3(a, b, c, set_labels=labels)
        

def _venn_diagram3(a, b, c, set_labels=None):
    
    a = list(set(a))
    b = list(set(b))
    c = list(set(c))
    
    only_a = len( [x for x in a if x not in b+c] )
    only_b = len( [x for x in b if x not in a+c] )
    only_c = len( [x for x in c if x not in a+b] )

    a_b = len(np.intersect1d(a, b))
    a_c = len(np.intersect1d(a, c))
    b_c = len(np.intersect1d(b, c))
    
    a_b_c = len([ x for x in a if (x in b) and (x in c)])

    venn3(subsets=(only_a, only_b, a_b, only_c, a_c, b_c, a_b_c), 
          set_labels=set_labels)
    

def _venn_diagram2(a, b, set_labels=None):
    
    a = list(set(a))
    b = list(set(b))
    
    only_a = len( [x for x in a if x not in b] )
    only_b = len( [x for x in b if x not in a] )

    a_b = len(np.intersect1d(a, b))
    
    venn2(subsets=(only_a, only_b, a_b), 
          set_labels=set_labels)
    