import collections


def scale_zscore(df, standardize_stats):
    df2 = df.copy()


    no_normalize = ["ids", "mmsi", "time", "from_locode", "to_locode", "combi", "type"]
    normalize = df2.columns[~df2.columns.isin(no_normalize)]
    for column in df2[normalize].columns:
        df2[column] = (
            df2[column] - standardize_stats[column]["mean"]
        ) / standardize_stats[column]["std"]
    return df2


def scale_minmax(df, standardize_stats):
    df2 = df.copy()
    

    no_normalize = ["ids", "mmsi", "time", "from_locode", "to_locode", "combi", "type"]
    normalize = df2.columns[~df2.columns.isin(no_normalize)]
    for column in df2[normalize].columns:
        df2[column] = (df2[column] - standardize_stats[column]["min"]) / (
            standardize_stats[column]["max"] - standardize_stats[column]["min"]
        )
    return df2


def scale_zscore_minmax(df,zscore_stats=None,min_max_stats=None):
    df2 = df.copy()

    if zscore_stats==None:
        zscore_stats = df2.describe()
    df_zscore = scale_zscore(df2,zscore_stats)


    if min_max_stats==None:
        min_max_stats = df_zscore.describe()
        
    df_minmax = scale_minmax(df_zscore, min_max_stats)

    result = collections.namedtuple("result", ["df", "zscore_stats", "minmax_stats"])
    res = result(df_minmax, zscore_stats, min_max_stats)

    return res
