import numpy as np
import pandas as pd
import scipy.stats as st



def get_accuracy(length, predicted_length_inv, width, predicted_width_inv):
    length_bool_99CI = np.array([length[i] in range(predicted_length_inv[i][0],predicted_length_inv[i][-1]) for i in range(len(length)) ])*1
    length_bool_75CI = np.array([length[i] in range(predicted_length_inv[i][1],predicted_length_inv[i][-2]) for i in range(len(length)) ])*1

    length_acc_99_CI = (np.sum(length_bool_99CI)+0)/(np.sum(length_bool_99CI)+0+0+(len(length_bool_99CI)-np.sum(length_bool_99CI)))
    length_acc_75_CI = (np.sum(length_bool_75CI)+0)/(np.sum(length_bool_75CI)+0+0+(len(length_bool_75CI)-np.sum(length_bool_75CI)))

    width_bool_99CI = np.array([width[i] in range(predicted_width_inv[i][0],predicted_width_inv[i][-1]) for i in range(len(length)) ])*1
    width_bool_75CI = np.array([width[i] in range(predicted_width_inv[i][1],predicted_width_inv[i][-2]) for i in range(len(length)) ])*1

    width_acc_99_CI = (np.sum(width_bool_99CI)+0)/(np.sum(width_bool_99CI)+0+0+(len(width_bool_99CI)-np.sum(width_bool_99CI)))
    width_acc_75_CI = (np.sum(width_bool_75CI)+0)/(np.sum(width_bool_75CI)+0+0+(len(width_bool_75CI)-np.sum(width_bool_75CI)))

    return [length_acc_99_CI,length_acc_75_CI], [width_acc_99_CI,width_acc_75_CI]


def evaluate(model, images, metadata, targets):

    assert metadata.shape[-1] == 2, "only cog and sog are supported thus far."
    scores = model.evaluate([images, metadata], targets, verbose=0)
    return scores


def get_abs_errors(length, predicted_length_inv, width, predicted_width_inv):
    try:
        abs_errors_length = abs(abs(length) - abs(predicted_length_inv))
        abs_errors_width = abs(abs(width) - abs(predicted_width_inv))
        abs_errors_length_normalised = abs_errors_length / abs_errors_length.max()
        abs_errors_width_normalised = abs_errors_width / abs_errors_width.max()
        normalized_errors = (abs_errors_length_normalised+abs_errors_width_normalised)/2
        normalized_errors = normalized_errors/normalized_errors.max()

    except Exception as e:
        print("error in finding abs errors")
        pass
    return abs_errors_length, abs_errors_width, normalized_errors


def get_rmse_errors(abs_errors_length, abs_errors_width):
    try:
        rmse_length = np.sqrt(np.mean((abs_errors_length**2), axis=0))
        rmse_width = np.sqrt(np.mean((abs_errors_width**2), axis=0))
    except Exception as e:
        print("error in finding rmse errors")
        pass
    return rmse_length, rmse_width


def get_mse_errors(abs_errors_length, abs_errors_width):
    try:
        mse_length = np.mean((abs_errors_length**2), axis=0)
        mse_width = np.mean((abs_errors_width**2), axis=0)

    except Exception as e:
        print("error in finding mse errors")
        pass
    return mse_length, mse_width


def get_mean_abs_error_size_distribution(
    length, abs_errors_length, width, abs_errors_width, nbins: int = 100
):
    d = np.array([length, abs_errors_length, width, abs_errors_width]).T
    numbins = nbins
    bins_l = np.linspace(d[:, 0].min(), d[:, 0].max(), numbins)
    inds_l = np.digitize(d[:, 0], bins_l)

    bins_w = np.linspace(d[:, 2].min(), d[:, 2].max(), numbins)
    inds_w = np.digitize(d[:, 2], bins_w)

    d = np.array(
        [
            length,
            abs_errors_length,
            inds_l,
            width,
            abs_errors_width,
            inds_w,
        ]
    ).T
    df = pd.DataFrame(
        d, columns=["length", "error_l", "bin_l", "width", "error_w", "bin_w"]
    )
    errors_l = df.groupby("bin_l").mean()
    errors_w = df.groupby("bin_w").mean()
    df.bin_l = df.bin_l.astype("int")
    df.bin_w = df.bin_w.astype("int")

    ci_l = []
    for i in range(len(df.bin_l.unique())):
        bin = df.bin_l.unique()[i]
        ci = st.norm.interval(
            alpha=0.95,
            loc=np.mean(df.error_l[df.bin_l == bin].values),
            scale=st.sem(df.error_l[df.bin_l == bin].values),
        )
        if np.isnan(ci).any():
            ci = (
                df[df.bin_l == bin].error_l.values[0],
                df[df.bin_l == bin].error_l.values[0],
            )
        ci_l.append((bin, ci[0], ci[1]))

    # create 95% confidence interval for population mean weight
    ci_l = np.array(ci_l, dtype=object)
    dfci_l = pd.DataFrame(ci_l, columns=["bins", "ci_lower", "ci_upper"])
    errors_l = errors_l.join(dfci_l.set_index("bins"), on="bin_l")
    errors_l.ci_lower = errors_l.ci_lower.astype("float")
    errors_l.ci_upper = errors_l.ci_upper.astype("float")

    ci_w = []
    for i in range(len(df.bin_w.unique())):
        bin = df.bin_w.unique()[i]
        ci = st.norm.interval(
            alpha=0.95,
            loc=np.mean(df.error_w[df.bin_w == bin].values),
            scale=st.sem(df.error_w[df.bin_w == bin].values),
        )
        if np.isnan(ci).any():
            ci = (
                df[df.bin_w == bin].error_w.values[0],
                df[df.bin_w == bin].error_w.values[0],
            )

        ci_w.append((bin, ci[0], ci[1]))
    # create 95% confidence interval for population mean weight
    ci_w = np.array(ci_w, dtype=object)
    dfci_w = pd.DataFrame(ci_w, columns=["bins", "ci_lower", "ci_upper"])
    errors_w = errors_w.join(dfci_w.set_index("bins"), on="bin_w")
    errors_w.ci_lower = errors_w.ci_lower.astype("float")
    errors_w.ci_upper = errors_w.ci_upper.astype("float")
    errors_l = errors_l
    errors_w = errors_w

    return errors_l, errors_w
