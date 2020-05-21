import numpy as np
import pandas as pd
from glob import glob
from pyteomics import mzxml


class example_files():
    def __init__(self):
        self._mzxml = sorted(list(glob('/home/swacker/workspace/uofc/data/metabolomics/ms1/mzxml/*')))
        self._parquet = sorted(list(glob('/home/swacker/workspace/uofc/data/metabolomics/ms1/parquet/*')))

    @property
    def mzxml(self):
        return self._mzxml

    @property
    def parquet(self):
        return self._parquet


def mzxml_to_pandas_df(filename):
    slices = []
    file = mzxml.MzXML(filename)
    print('Reading:', filename)
    while True:
        try:
            slices.append(pd.DataFrame(file.next()))
        except:
            break
    df = pd.concat(slices)
    df_to_numeric(df)
    df['intensity array'] = df['intensity array'].astype(np.float64)
    return df


def df_to_numeric(df):
    for col in df:
        df.loc[:, col] = pd.to_numeric(df[col], errors='ignore')


def slize_ms1_mzxml(df, rt, delta_rt, mz, delta_mz):
    slize = df.loc[(rt-delta_rt <= df.retentionTime) &
                   (df.retentionTime <= rt+delta_rt) &
                   (mz-delta_mz <= df['m/z array']) & 
                   (df['m/z array'] <= mz+delta_mz)]
    return slize

def plot_slize(slize, bins=200, cmin=0, cmax=1000):
    #scatter(slize['retentionTime'], slize['m/z array'], c=slize['intensity array'], s=0.1)
    #sns.jointplot('retentionTime', 'm/z array', data=slize, kind="kde", space=0, color="g")
    hist2d(slize['retentionTime'], slize['m/z array'],
           bins=bins, weights=slize['intensity array'].apply(np.log1p),
           cmin=cmin, cmax=cmax)
    
    xlabel('Retention Time [min]')
    ylabel('m/z')