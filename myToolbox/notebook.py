from IPython.core.display import display, HTML

def width(width):
    display(HTML("<style>.container { width:%d%% !important; }</style>" %width))