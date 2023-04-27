

import cartopy.crs as ccrs
import cartopy.crs as crs
import matplotlib.pyplot as plt


import cartopy
import matplotlib
from matplotlib import rc





def initi(size: int = 32):
    matplotlib.rcParams.update({"font.size": size})
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)



def plot_tracks(df):

    mrc = ccrs.Mercator()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())

    ax.stock_img()
    ax.gridlines(draw_labels=True)
    try:
        ax.scatter(df.long.to_numpy(), df.lat.to_numpy(), c="red", s=10)
    except:
        pass
    ax.coastlines()
    try:
        ax.set_extent(
            [df.long.min() - 1, df.long.max() + 1, df.lat.min() - 1, df.lat.max() + 1]
        )
    except:
        ax.set_extent(
            [df.lon.min() - 1, df.lon.max() + 1, df.lat.min() - 1, df.lat.max() + 1]
        )
    plt.show()



def plot_polygon(df):
    """
    Plotting polygons from a gdf.
    """

    projections = [
        cartopy.crs.PlateCarree(),
        cartopy.crs.Robinson(),
        cartopy.crs.Mercator(),
        cartopy.crs.Orthographic(),
        cartopy.crs.InterruptedGoodeHomolosine(),
    ]

    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=projections[0])
    try:
        extent = [
            df.long.min() - 1,
            df.long.max() + 1,
            df.lat.min() - 1,
            df.lat.max() + 1,
        ]
        ax.set_extent(extent)
    except:
        pass

    ax.stock_img()
    ax.coastlines(resolution="10m")
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor="black")
    ax.gridlines()
    try:
        ax.scatter(df.long.to_numpy(), df.lat.to_numpy(), c="red", s=10,alpha=0.3)
    except:
        pass



    gl3 = ax.gridlines(
        draw_labels=True, linewidth=1.0, color="black", alpha=0.75, linestyle="--"
    )
    gl3.left_labels = False
    gl3.top_labels = False
    gl3.xlabel_style = {"size": 10.0, "color": "gray", "weight": "bold"}
    gl3.ylabel_style = {"size": 10.0, "color": "gray", "weight": "bold"}
    gl3.xlabel_style = {"rotation": 45}
    gl3.ylabel_style = {"rotation": 45}

    return None
