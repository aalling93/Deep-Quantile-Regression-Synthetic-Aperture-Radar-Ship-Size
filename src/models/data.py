import glob
import pandas as pd
import numpy as np
from ._util import seed_everything



def load_metadata(path):


    imgs_data, imgs_shipdata, image_names = get_metadata(path)
    imgs_shipdata = clean_dataframe(imgs_shipdata)
    imgs_data = clean_dataframe(imgs_data)
    return imgs_data, imgs_shipdata, image_names

def get_metadata(path):
    """imgs_data is the metadata for the image (i.e. sentinel-1 take etc.)
    imgs_shipdata is the added data on the ships, i.e. length and width etc.
    """
    imgs_data = pd.read_csv(path + "imgs_data.csv")
    imgs_shipdata = pd.read_csv(path + "imgs_shipsdata.csv")
    image_names = glob.glob(path + "imgs/*.npy")
    return imgs_data, imgs_shipdata, image_names


def get_data_many_images(image_names, imgs_data, imgs_shipdata, imsize: int = 100):
    result = [
        get_data_one_image(imgs_data, imgs_shipdata, image_name=name, imsize=imsize)
        for name in image_names
    ]

    image = np.array(result[0])
    metadatum = np.array(result[1])
    target = np.array(result[2])
    uuid = np.array(result[2])
    return image, metadatum, target, uuid

def clean_dataframe(df):
    try:
        df['MMSI'] = df.mmsi
        del df['mmsi']
    except Exception as e:
        pass

    try:
        df['ais_cog'] = df.ais_bearing
        del df['ais_bearing']
    except Exception as e:
        pass

    try:
        df['ais_sog'] = df.ais_velocity
        del df['ais_velocity']
    except Exception as e:
        pass

    try:
        df['length'] = df.LENGTH
        del df['LENGTH']
    except Exception as e:
        pass

    try:
        df['width'] = df.BREDTH
        del df['BREDTH']
    except Exception as e:
        pass

    try:
        df['type'] = df.VESSEL_TYPE
        del df['VESSEL_TYPE']
    except Exception as e:
        pass

    

    return df


def get_data_one_image(
    imgs_data, imgs_shipdata, image_name: str = "", imsize: int = 100
):
    # getting the uuid of the iamge
    img_id = image_name.split("/")[-1].split(".npy")[0]

    metadatum = imgs_data[imgs_data.image_id == img_id]
    mmsi = metadatum.ais_mmsi.values[0]
    target = imgs_shipdata[imgs_shipdata.MMSI == mmsi]

    if (
        np.isnan(metadatum.ais_cog.values[0]) == False
        and np.isnan(metadatum.ais_sog.values[0]) == False
    ):
        try:
            image = np.load(image_name)
            startx = image.shape[0] // 2 - imsize // 2
            starty = image.shape[1] // 2 - imsize // 2

            target = np.array(
                [
                    int(target.length.values[0].split(" ")[0]),
                    int(target.width.values[0].split(" ")[0]),
                    target.type,
                ]
            )
            metadatum = np.array(
                [
                    metadatum.ais_mmsi.values[0],
                    metadatum.sar_offset.values[0],
                    metadatum.sar_incidence_angle.values[0],
                    metadatum.ais_cog.values[0],
                    metadatum.ais_sog.values[0],
                ]
            )
            image = image[
                starty : starty + imsize,
                startx : startx + imsize,:
            ]
            return image, metadatum, target, img_id
        except Exception as e:
            print(e)
            return None
