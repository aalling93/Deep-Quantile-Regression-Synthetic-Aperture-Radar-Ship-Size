import pandas as pd
import numpy as np
import collections


def resample_all_ids(df, resampling_frq: int = 5):
    df = df.copy()
    df = df.groupby("ids").apply(resample_ids, resampling_frq=resampling_frq)
    df = df.reset_index()
    try:
        del df["level_1"]
    except:
        pass
    try:
        del df["level_0"]
    except:
        pass
    return df



def get_datasets(
    df, lookback_offset: int = 1, lookback: int = 5, target_observations: int = 1
):

    sample = (
        df.groupby("ids")
        .apply(
            datasets_training,
            lookback_offset=lookback_offset,
            lookback=lookback,
            target_observations=target_observations,
        )
        .apply(list)
    )

    samples = np.array([sam[0] for sam in sample],dtype=object)
    targets = np.array([sam[1] for sam in sample],dtype=object)
    features = sample[0][2]



    return samples, targets, features


def datasets_training(
    df, lookback_offset: int = 1, lookback: int = 5, target_observations: int = 1
):
    samples = []
    targets = []
    for rows in range(
        lookback_offset, df.shape[0] - lookback - target_observations + 1
    ):
        samples.append(df.iloc[rows : rows + lookback, :].to_numpy())
        targets.append(
            df.iloc[
                rows + lookback : rows + lookback + target_observations, :
            ].to_numpy()
        )

    #
    return np.array(samples), np.array(targets), np.array(df.columns.values)



def resample_ids(df, resampling_frq: int = 5):
    """

    Timestamps do not have fixed intervals.
    It ranges from less than 10 sec to more than hours aparts (amount of hours defined by ais.df_split_ais() )
    To account for this large varibility, and to add more generability in the NN, a resampling is performed.
    This resampling resampel the data using a mean method.



    First, we generate the underlying data grid by using mean().
    This generates the grid with NaNs as values. Afterwards, we fill the NaNs with interpolated values by calling the interpolate() method on the read value column
    """
    df.index = df.time

    try:
        try:
            df = (
                df.resample(f"{resampling_frq}min")
                .mean()
                .interpolate(method="cubicspline")
            )

        except:
            try:
                df = (
                    df.resample(f"{resampling_frq}min")
                    .mean()
                    .interpolate(method="spline", order=3)
                )
            except:
                try:
                    df = (
                        df.resample(f"{resampling_frq}min")
                        .mean()
                        .interpolate(method="spline", order=3, s=0.0)
                    )
                except:
                    df = (
                        df.resample(f"{resampling_frq}min", origin="start").mean().pad()
                    )

                pass
            pass
        df = df.reset_index()
    except Exception as e:
        print(f"Error in resampling id: {df.ids.iloc[0]}: {e}")
        pass

    return df


def raw2clean(
    df_raw, t_messages=100, t_pausetime=1000, f_resampling=10, t_maxlengt=300, verbose=0
):
    """

    t_messages = 100 # threshold for number of messages
    t_pausetime = 1000 #threshold for pause time
    f_resampling = 10 #resampling frequence
    t_minlengt = 10 # minimum length of sequences.
    t_maxlengt = 300 # maximum length of sequences. sequence time = f_resampling*samples.


    """
    cargo_df = df_for_shiptype(df_raw, t_messages)
    if verbose > 0:
        print(len(cargo_df))
    del df_raw

    cargo_df_split = df_split_ais(cargo_df, t_pausetime)
    if verbose > 0:
        print(len(cargo_df_split))
    del cargo_df

    cargo_df_split_resampl = df_resampling(cargo_df_split, f_resampling)
    if verbose > 0:
        print(len(cargo_df_split_resampl))
    del cargo_df_split

    cargo_df_split_resampl_maxlength = df_splt_smaller_seq(
        cargo_df_split_resampl, t_maxlengt
    )
    if verbose > 0:
        print(len(cargo_df_split_resampl_maxlength))
    del cargo_df_split_resampl

    cargo_df_split_resampl_maxlength_corrected = df_correction(
        cargo_df_split_resampl_maxlength
    )
    if verbose > 0:
        print(len(cargo_df_split_resampl_maxlength_corrected))
    del cargo_df_split_resampl_maxlength

    testing_set = df_to_numpy_training(cargo_df_split_resampl_maxlength_corrected)

    if verbose > 0:
        print(testing_set.shape)

    return testing_set, cargo_df_split_resampl_maxlength_corrected


def make_dataset_single(
    df, lookback: int = 5, lookback_offset: int = 0, target_observations: int = 0
):
    samples = []
    targets = []
    for rows in range(
        lookback_offset, df.shape[0] - lookback - target_observations + 1
    ):
        samples.append(df.iloc[rows : rows + lookback, :].to_numpy())
        targets.append(
            df.iloc[
                rows + lookback : rows + lookback + target_observations, :
            ].to_numpy()
        )
    return samples, targets


def add_datasets_all(df):
    try:
        samples, targets = df.groupby("ids").apply(make_dataset_single)
        return samples, targets
    except:
        pass


def get_index(
    samples_description: list,
    parms=[
        "lat",
        "long",
        "sog",
        "cog",
        "Total_distance",
        "Running_Distance",
        "delta_lat",
        "delta_long",
    ],
):
    features = samples_description.isin(parms)
    features = pd.Series(features)
    features = features[features].index

    return features
