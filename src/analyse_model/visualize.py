import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#
COLORMAP = "turbo"
IMG_COLORMAP = "gray"
STYLE = "default"
STYLE = "seaborn-colorblind"


def initi(size: int = 32):
    """Looking like latex"""
    matplotlib.rcParams.update({"font.size": size})
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)


def scatter_xy_c(
    x,
    y,
    c,
    x_name: str = "",
    y_name: str = "",
    c_name: str = "",
    save_name: str = "",
    save=False,
):
    try:
        with plt.style.context((STYLE,)):
            plt.figure(figsize=(15, 13))
            x2 = np.array([x.min(), x.max()])
            y2 = np.array([x.min(), x.max()])
            im = plt.scatter(
                np.squeeze(x),
                y,
                c=c,
                cmap=COLORMAP,
            )
            plt.plot(x2, y2, lw=2)
            plt.colorbar(im, label=f"{c_name}")
            plt.ylabel(f"{y_name}")
            plt.xlabel(f"{x_name}")
            if save == True:
                plt.savefig(
                    f"{save_name}.png",
                    dpi=360,
                    bbox_inches="tight",
                    pad_inches=1,
                    transparent=True,
                )
            # plt.show()
    except:
        pass


def length_velocity(
    predicted_length_inv, length, velocity, save_name: str = "", save=False
):
    scatter_xy_c(
        predicted_length_inv,
        length,
        velocity,
        x_name="Estiamted length (m)",
        y_name="True length (m)",
        c_name="Velocity (kn)",
        save_name=save_name,
        save=save,
    )


def width_velocity(
    predicted_width_inv, width, velocity, save_name: str = "", save=False
):
    scatter_xy_c(
        predicted_width_inv,
        width,
        velocity,
        x_name="Estiamted width (m)",
        y_name="True width (m)",
        c_name="Velocity (kn)",
        save_name=save_name,
        save=save,
    )


def length_abs_error(
    predicted_length_inv, length, abs_length_errors, save_name: str = "", save=False
):
    scatter_xy_c(
        predicted_length_inv,
        length,
        abs_length_errors,
        x_name="Estiamted length (m)",
        y_name="True length (m)",
        c_name="Abs error (m)",
        save_name=save_name,
        save=save,
    )


def width_abs_error(
    predicted_width_inv, width, abs_width_errors, save_name: str = "", save=False
):
    scatter_xy_c(
        predicted_width_inv,
        width,
        abs_width_errors,
        x_name="Estiamted width (m)",
        y_name="True width (m)",
        c_name="Abs error (m)",
        save_name=save_name,
        save=save,
    )


def width_cog(
    abs_width_errors, cog, predicted_width_inv, save_name: str = "", save=False
):
    scatter_xy_c(
        abs_width_errors,
        cog,
        predicted_width_inv,
        xy_name="Bearing",
        xy_unit="(deg)",
        c_name="Abs error (m)",
        save_name=save_name,
        save=save,
    )


def plot_mean_abs_error_size_distribution(
    errors_l, errors_w, self, length, width, save_name: str = "", save=False
):
    with plt.style.context((STYLE,)):
        plt.figure(figsize=(10, 10))
        axes1 = plt.gca()
        axes2 = axes1.twinx()

        axes3 = axes1.twiny()
        axes4 = axes2.twiny()

        ln1 = axes1.plot(
            errors_l.error_l, errors_l.length, color="black", label="Length"
        )
        axes1.fill_betweenx(
            errors_l.length.values,
            errors_l.ci_lower.values - 0.0001,
            errors_l.ci_upper.values + 0.0001,
            alpha=0.4,
            color="black",
        )
        ln2 = axes2.plot(errors_w.error_w, errors_w.width, c="red", label="width")
        axes2.fill_betweenx(
            errors_w.width.values,
            errors_w.ci_lower.values - 0.0001,
            errors_w.ci_upper.values + 0.0001,
            alpha=0.4,
            color="red",
        )
        # plt.xlabel('Mean error (m)')

        ln3 = axes3.hist(
            length,
            orientation="horizontal",
            lw=2,
            edgecolor=(0, 0, 0, 0.5),
            facecolor=(0, 0, 0, 0.1),
            label="Length distribution",
        )
        lnshist1 = [l.get_label() for l in ln3[2]][0]

        ln4 = axes4.hist(
            width,
            orientation="horizontal",
            lw=2,
            ls="dashed",
            edgecolor=(1, 0, 0, 0.5),
            facecolor=(1, 0, 0, 0.1),
            label="Width distribution",
        )
        lnshist2 = [l.get_label() for l in ln4[2]][0]

        axes1.set_ylabel("Length of ships (m)")
        axes2.set_ylabel("Width of ships (m)")

        # added these three lines
        lns = ln1 + ln2
        # plots = ln1+ln2 + ln3[2] + ln4[2]
        labs = [l.get_label() for l in lns]
        lnshist = [lnshist1, lnshist2]
        labs = labs + lnshist

        # right, left, top, bottom
        axes4.spines["top"].set_position(("outward", 80))

        axes4.xaxis.label.set_color("red")

        axes1.legend(lns, labs, loc=0)
        axes1.set_xlabel("Mean error of estiamates (m)")
        axes3.set_xlabel("Length distribution of ships")
        axes4.set_xlabel("Width distribution of ships")
        axes4.tick_params(axis="x", colors="red")
        axes2.tick_params(axis="y", colors="red")
        axes2.yaxis.label.set_color("red")
        # axes4.xaxis.set_ticks([])
        if save == True:
            plt.savefig(f"{save_name}.png", dpi=360, bbox_inches="tight", pad_inches=1)
        # plt.show()


def distribution(
    avgs,
    true_value,
    x_name: str = "",
    save_name: str = "",
    save=False,
    hist_vals: list = [-100, 700, 20],
):
    """
    avgs: array
    """
    with plt.style.context((STYLE,)):
        plt.figure(figsize=(12, 10))
        density = stats.gaussian_kde(avgs.astype(int))
        n, x, _ = plt.hist(
            avgs.astype(int),
            bins=np.linspace(hist_vals[0], hist_vals[1], hist_vals[2]),
            histtype="step",
            density=True,
        )
        plt.axvline(x=true_value, color="k", label=f"{x_name}")
        plt.plot(x, density(x))
        if save == True:
            plt.savefig(
                f"{save_name}.png",
                dpi=360,
                bbox_inches="tight",
                pad_inches=1,
                transparent=True,
            )
        # plt.show()


def single_ship_cross(image):
    plt.imshow(image[:, :, 1], cmap=IMG_COLORMAP)
    plt.colorbar()


def single_ship_co(image):
    plt.imshow(image[:, :, 0], cmap=IMG_COLORMAP)
    plt.colorbar()
