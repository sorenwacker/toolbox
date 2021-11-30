import numpy as np

def col_to_class(df, col_name, possible_values=None, delete_col=True):
    print('Converting %s to classes' %col_name)
    tmp = df.loc[:, []].copy()
    if possible_values is None:
        possible_values = df[col_name].value_counts().index
    for value in possible_values:
        new_col_name = col_name+'_'+str(value)
        tmp.loc[:, new_col_name] = (df.loc[:, col_name] == value).astype(int)
    tmp = tmp[sorted(tmp.columns)]
    if delete_col:
        df.drop(col_name, 1, inplace=True)
    return df.join(tmp)   

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def get_dublicate_col_values(df, col_name):
    val_counts = df[col_name].value_counts()
    values = val_counts[val_counts > 1].index
    dublicates = df[df[col_name].isin(values)].sort_values(col_name)
    return dublicates

def sort_df_by_row_count(df, axis=1, ascending=True):
    ndx = df.sum(axis=axis).sort_values(ascending=ascending).index
    return df.reindex(ndx, axis=(axis+1) % 2)

def stratify_df(df, columns, n_sample=None, random_state=None):
    count_per_group = df.groupby(columns).count().iloc[:,0]
    if n_sample is None:
        n_sample = count_per_group.min().min()
        print(f'Using n_sample={n_sample}.')
    stratified = df.groupby(columns, group_keys=False).apply(lambda x: x.sample(min(len(x), n_sample, random_state=random_state)))
    return stratified
