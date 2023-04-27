
import numpy as np
import pandas as pd
from ._added_value import *




def remove_min_messages(df,min_messages:int=10):
    #df = df.copy()
    try:
        delete_id = []
        for id in df.ids.unique():
            if len(df[df.ids==id])<min_messages:
                delete_id.append(id)
        df = df[~df.ids.isin(delete_id)] 
        return df
    except Exception as e:
        print(e)
        pass




def min_seqlength(df, lookback=30, loockback_multiples=1, verbose=0):
    """
    remove all
    """
    df_out = []
    for i in range(len(df)):
        if len(df[i]) > (lookback * loockback_multiples):
            df_out.append(df[i])

    df_out = np.array(df_out)

    if verbose > 0:
        print(f"Number of sequences: {len(df_out)}. ")

    return df_out


def split_trajecotries(
    df, allowed_stop: float = 100, min_messages: int = 120, time_id: str = ""
):
    """
    This function spilt up a AIS df if there is a brek of
    Since large datasets with long periods will be used, it is important to split up tracks where a ship has been in e.g a habour for a few weeks.
    There can be multiple of such anchor points.
    It is the individual tracks that is of interest, and not a ships lifelong journey.

    allowed_stop_time[int]: Number of minuts alloowed to break.
                            if time between messages are larger than allowed_stop_time, the df will be split up.

    If allowed_stop_time is large: only longer stops are used to split up the df.
    If allowed_stop_time is low: only frequencly recorded tracks will be used. This will reduce the uncertainty when training.
    For long term prediction, a large allowed_stop_time is ok.
    If the task is to predict short term ship tracks, a small allowed_stop_time is needed.

    """

    try:
        df_temp = df.groupby("imo").apply(
            split_trajectory, allowed_stop=allowed_stop, time_id=time_id
        )
        df_temp = df_temp.reset_index(drop=True)
    except:
        try:
            df_temp = df.groupby("mmsi").apply(
                split_trajectory, allowed_stop=allowed_stop, time_id=time_id
            )
            df_temp = df_temp.reset_index(drop=True)
        except:
            df_temp = df.groupby("ids").apply(
                split_trajectory, allowed_stop=allowed_stop, time_id=time_id
            )
            df_temp = df_temp.reset_index(drop=True)

    try:
        delete_id = []
        for id in df_temp.ids.unique():
            if len(df_temp[df_temp.ids == id]) < min_messages:
                delete_id.append(id)

        df_temp = df_temp[~df_temp.ids.isin(delete_id)]
    except:
        pass

    return df_temp


def split_trajectory(df, allowed_stop: int = 100, time_id: str = ""):
    """ """
    df_combined = pd.DataFrame()
    try:
        position_split = df.query(
            f"running_time > {allowed_stop} or running_time < 0"
        ).index.values.tolist()
    except Exception as e:
        print(e)
    try:
        if len(position_split) > 0:
            position_split.insert(0, 0)
            position_split.insert(len(position_split), len(df))
            for split in range(len(position_split) - 1):
                df_temp = df.iloc[position_split[split] : position_split[split + 1], :]
                df_temp = df_temp.copy()
                df_temp["ids"] = f"{abs(df.mmsi.iloc[0])}_{split+1}"
                #df_temp.loc[0,["runing_time"]] = 0
                #df_temp.loc[0,["runing_distance"]] = 0
                df_combined = pd.concat([df_combined, df_temp])
            del df_temp
        else:
            df_combined = df
            df_combined["ids"] = f"{abs(df.mmsi.iloc[0])}_{1}"

    except Exception as e:
        print("error in split_trajectory: ", e)
        pass

    return df_combined






def load_datasets(train, val, test):

    training_df = load_dataset(train)
    validation_df = load_dataset(val)
    testing_df = load_dataset(test)

    return training_df, validation_df, testing_df


def get_tracks_mmsi(df, mmsi: list):
    try:
        df = df[df.mmsi.isin(mmsi)]
        return df
    except:
        print("\nCan't find MMSIs.")


def get_tracks_ids(df, ids: list = []):
    try:
        df = df[df.ids.isin(ids)]
        return df
    except:
        print("\nCan't find ids.")


def load_dataset(df):
    try:
        df = pd.read_pickle(df)
        df = df.dropna()
    except:
        try:
            df = pd.read_csv(df)
            df = df.dropna()
        except:
            try:
                df = pd.read_feather(df)
                df = df.dropna()
            except:
                try:
                    df = df
                    df = df.dropna()
                except Exception as e:
                    print(e)

        

    try:
        df = df.rename(columns={"datatime": "time"})
    except:
        pass

    try:
        df = df.rename(columns={"lon": "long"})
    except:
        pass
    try:
        df.mmsi = df.MMSI
        del df["MMSI"]
    except:
        pass

    try:
        df.cog = df.cog.astype(np.float32)
        df.sog = df.sog.astype(np.float32)
        df.lat = df.lat.astype(np.float32)
        df.long = df.long.astype(np.float32)
        # df.long = df.long+180
        # df.lat = df.lat+90
        df.mmsi = df.mmsi.astype(np.int16)
        df.imo = df.imo.astype(np.int16)
    except:
        pass

    return df
