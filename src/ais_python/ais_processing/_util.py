def save_ais_df(df, filename: str = "ais", ext: str = "pkl"):
    """ """
    if ext.lower() == "feather":
        df.reset_index().to_feather(f"{filename}.feather")
    elif ext.lower() == "pkl" or ext.lower() == "pickle":
        df.reset_index().to_pickle(f"{filename}.pkl")
    elif ext.lower() == "csv":
        df.reset_index().to_csv(f"{filename}.csv")
    else:
        df.reset_index().to_feather(f"{filename}.feather")
        df.reset_index().to_pickle(f"{filename}.pkl")
        df.reset_index().to_csv(f"{filename}.csv")
