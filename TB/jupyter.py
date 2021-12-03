from IPython.core.display import display, HTML


def notebook_width(width):
    display(HTML("<style>.container { width:%d%% !important; }</style>" % width))
