import numpy as np
from matplotlib_venn import venn3

def venn_diagram(a, b, c, set_labels=['A', 'B', 'C']):
    
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