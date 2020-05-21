import pandas as pd
import numpy as np
import os, re

from os.path import isdir, isfile, basename

from os.path import basename, dirname
from glob import glob
from pathlib import Path as P
from tqdm.notebook import tqdm

import seaborn as sns

# STOP
from .jupyter import *
from .general import *
from .pd import *

with open(__file__, 'r') as this_file:
    for line in this_file.readlines():
        if re.search('STOP', line):
            break
        print(line, end="")
