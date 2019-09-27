import pandas as pd
from pyteomics import mzxml
from glob import glob

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
    df['intensity array'] = df['intensity array'].astype(float64)
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
