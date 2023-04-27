from . import _ais_helper_funcs
from .data_cleaning import *
from ._added_value import *
from ._util import save_ais_df

import datetime
import pandas as pd

class Ais_processing:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def load_ais(self, path):
        """'
        Loading an AIS file from either fether, pickle or csv format.
        AIS data should include
        mmsi, time, lat, long, sog and cog.
        """
        self.df = _ais_helper_funcs.load_dataset(path)

    def clean_ais(self, **kwargs):
        """'
        Cleaning the AIS data
        1) remove Nan
        2) Rounding data (defined by args)
        3) filter data, defined by kwargs
        """
        self.df = remove_nan(self.df)
        self.df = round_data(self.df, **kwargs)
        self.df = filter_lat(self.df, **kwargs)
        self.df = filter_long(self.df, **kwargs)
        self.df = filter_cog(self.df, **kwargs)
        self.df = filter_sog(self.df, **kwargs)

    def get_trajectories(self, allowed_stop: int = 100, min_messages: int = 10):
        """'
        Splitting the entire dataframe into trajectories.
        These trajectories are found for each ship.
        For each ship, a subtrajectory is found by splitting the data after a time period
        Should also add, e.g., harbours..
        """
        self.df = add_ids_all(
            self.df, allowed_stop=allowed_stop, min_messages=min_messages
        )

    def add_derived_info(self):
        """adding derived info"""
        self.df = add_derived_all(self.df)

    def save_df(self, filename: str = "", ext: str = "pkl"):
        save_ais_df(self.df, filename=filename, ext=ext)





def interpolate(df: pd.core.frame.DataFrame,freq:int=1):
    'df grouped by, e.g., MMSI or IDS. frq in minutes. '
    df.index = df.time
    df = df.resample(f"{freq}min").mean().interpolate(method="cubicspline").astype({"mmsi": int}).reset_index()
    return df

def interpolate_df(df: pd.core.frame.DataFrame,by:str='ids',freq:int=1):
    df = df.groupby(by).apply(interpolate, freq=freq).droplevel(1).reset_index()
    return df



def get_nearest_time(df: pd.core.frame.DataFrame, s1_time: datetime.datetime):
    df.index = df.time
    return df.iloc[df.index.get_indexer([s1_time], method="nearest")].reset_index(
        drop=True
    )


def get_nearest_time_df(
    df: pd.core.frame.DataFrame,
    s1_time: datetime.datetime = datetime.datetime(2019, 3, 30, 10, 5, 8),
    by: str = "ids",
):
    df = (
        df.groupby(by)
        .apply(get_nearest_time, s1_time=s1_time)
        .droplevel(1)
        .reset_index(drop=True)
    )
    return df