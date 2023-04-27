import numpy as np
import pandas as pd
from ._added_value_util import *
from ._ais_helper_funcs import *



def add_ids_all(df: pd.core.frame.DataFrame, allowed_stop: int = 350, min_messages: int = 10):
    df = df.groupby("mmsi").apply(
        get_ids, allowed_stop=allowed_stop, min_messages=min_messages
    )
    df = df.reset_index(drop=True)
    return df


def get_ids(df: pd.core.frame.DataFrame, allowed_stop: int = 350, min_messages: int = 10):
    df = get_time_spent_single(df)
    df = split_trajectory(df, allowed_stop=allowed_stop)
    df = remove_min_messages(df, min_messages=min_messages)
    return df


def get_derived_ids(df: pd.core.frame.DataFrame):
    df = get_total_distance_single(df)
    df = get_timeleft_single(df)
    df = get_total_time_spent_single(df)
    df = get_dist_pct_single(df)
    df = get_speed_single(df)
    df = get_bearing_single(df)
    df = get_co_ordinates_single(df)

    return df


def add_derived_all(df: pd.core.frame.DataFrame):
    df = df.groupby("ids").apply(get_derived_ids)
    df = df.reset_index(drop=True)
    return df


def get_bearing_single(df: pd.core.frame.DataFrame):
    """Adding calculated bearing to df"""
    df = df.set_index("lat")
    df.loc[:, "bearing_calculated"] = (
        df["long"]
        .rolling(2)
        .apply(calculate_bearing_single, raw=False)
        .astype(np.float32)
    )

    df = df.reset_index()
    df.loc[0, "bearing_calculated"] = df["bearing_calculated"].iloc[1]
    return df


def get_co_ordinates_single(df: pd.core.frame.DataFrame):
    """adding change in lat and long to df"""
    df = df.copy()
    df.loc[:, "delta_lat"] = df.lat.diff()
    df.loc[0, "delta_lat"] = 0

    df.loc[:, "delta_long"] = df.long.diff()
    df.loc[0, "delta_long"] = 0
    return df


def get_total_time_spent_single(df: pd.core.frame.DataFrame):
    """total time spent for single df with single mmsi or single ids."""
    try:
        df["total_time"] = df["running_time"].cumsum().astype(np.float32)
        return df
    except:
        return df


def get_timeleft_single(df: pd.core.frame.DataFrame):
    """time left (for e.g. ETA) spent for single df with single mmsi or single ids."""
    try:
        df["time_left"] = np.round(
            (df.to_time - df.time).astype("timedelta64[s]") / 60, 2
        )
        return df
    except:
        df["time_left"] = np.round(
            (df.time.iloc[-1] - df.time).astype("timedelta64[s]") / 60, 2
        )
        return df


def get_speed_single(df: pd.core.frame.DataFrame):
    """speed calculated for single df"""
    # runnin distances (nautilus miles) to meters
    # m/s to knots (by multiplying 1.943844)
    df = df.copy()
    df.loc[:, "speed_calculated"] = (
        (df.running_distance * 1852) / (df.running_time * 60) * 1.943844
    ).astype(np.float32)

    df["speed_calculated"] = df["speed_calculated"].round(4)
    df = df.reset_index(drop=True)
    df.loc[0, "speed_calculated"] = df.loc[1, "speed_calculated"]
    return df


def get_dist_pct_single(df: pd.core.frame.DataFrame):
    """dist travelled, pct for df with single mmsi or single ids."""
    try:
        df["distance_percentage"] = (
            df["total_distance"] / df["total_distance"].max()
        ).astype(np.float32)
    except:
        pass
    return df


def get_time_spent_single(df:pd.core.frame.DataFrame): 
    """running time for single df with single mmsi or single ids."""
    df = df.copy()
    try:
        temp = df.time.diff()
        temp.iloc[0] = pd.Timedelta(np.timedelta64(0, "ms"))
        df["running_time"] = temp
        df.running_time = df.running_time / np.timedelta64(1, "s")
        df.running_time = df.running_time / 60
        df = df.reset_index(drop=True)
        del temp
        return df
    except:
        return df


def get_total_distance_single(df: pd.core.frame.DataFrame):
    """total distance spent for single df with single mmsi or single ids."""
    df = df.copy()
    try:
        df = df.set_index("lat")

        # df["running_distance"] =  (df.rolling(2).apply(haversine_distance)
        df["running_distance"] = (
            df["long"].rolling(2, axis=0).apply(haversine_distance, raw=False)
        )
        df["total_distance"] = df["running_distance"].cumsum().astype(np.float32)
        df = df.reset_index(level=0)
        df.at[0, "total_distance"] = int("0")
        df.at[0, "running_distance"] = 0
    except Exception as e:
        print(e)

    return df
