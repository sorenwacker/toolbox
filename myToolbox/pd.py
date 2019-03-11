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
