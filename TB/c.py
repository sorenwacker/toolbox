import os
import re
import shutil
import logging

import pandas as pd
import numpy as np

from datetime import date
from os.path import isdir, isfile, basename, dirname, join
from time import sleep
from glob import glob
from pathlib import Path as P
from tqdm.notebook import tqdm

import seaborn as sns
import plotly.express as px

from matplotlib import pyplot as plt
import matplotlib as mpl
import plotly.express as px

from .plotting import savefig as sf
from .plotting import (
    plot_roc,
    plot_random_roc,
    plot_diagonal,
    plot_hlines,
    plot_vlines,
    heatmap,
    legend_outside,
    plot_dendrogram,
)


plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["figure.dpi"] = 150

pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

sns.set_context("paper")
sns.set_style("darkgrid")


def today():
    return date.today().strftime("%y%m%d")


def log2p1(x):
    try:
        return np.log2(x + 1)
    except:
        return x


remove_digits = lambda x: "".join([i for i in x if not i.isdigit()])

# STOP


with open(__file__, "r") as this_file:
    for line in this_file.readlines():
        if re.search("STOP", line):
            break
        print(line, end="")

logging.info(os.getcwd())
