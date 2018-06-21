from IPython.core.display import display, HTML

def widths(width):
    display(HTML("<style>.container { width:%s%% !important; }</style>", %width))