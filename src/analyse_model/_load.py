import pandas as pd
import glob

def load_data(data: list = []):
    """list of inputs:
    [images,metadata,targets]
    """
    images = data[0]
    metadata = data[1]
    targets = data[2]

    return images, metadata, targets

def load_history(model):
    # loading history, or trying to.
    try:
        hist_path_name = glob.glob(f"models/{model.name}/history_newest_epoch*")
        history = pd.read_pickle(hist_path_name[0])
        print("(Note) History of model training session loaded.\n")
    except Exception as e:
        print(f'Model history not loaded: {e}')
        pass
    return history




def load_scaling_values(list: list = []):
    
    

    return list[0],list[1],list[2],list[3],list[4],list[5],list[6],list[7],list[8],list[9],list[10],list[11],
    


def load_scaling(
    list:list=[]
):
    """scaling factor. min [inputs,meta,targets] , max[inputs,meta,targets]"""
    minimum_values = list[0]
    metadata_training_minimum_values = list[1]
    minimum_values_targets = list[2]
    maximum_values = list[3]
    metadata_training_maximum_values = list[4]
    maximum_values_targets = list[5]
    mean_values = list[6]
    metadata_training_mean_values = list[7]
    mean_values_targets = list[8]
    std_values = list[9]
    metadata_training_std_values = list[10]
    std_values_targets = list[11]

    index = ["images", "metadata", "targets"]

    minn = [
        minimum_values,
        metadata_training_minimum_values,
        minimum_values_targets,
    ]
    maxx = [
        maximum_values,
        metadata_training_maximum_values,
        maximum_values_targets,
    ]
    if len(mean_values) > 0:
        try:
            means = [
                mean_values,
                metadata_training_mean_values,
                mean_values_targets,
            ]
            stds = [
                std_values,
                metadata_training_std_values,
                std_values_targets,
            ]
        except:
            pass

    try:
        dict = {"minimum": minn, "maximum": maxx, "means": means, "stds": stds}
        normalisation_df = pd.DataFrame(dict, index=index)
    except:
        dict = {"minimum": minn, "maximum": maxx}
        normalisation_df = pd.DataFrame(dict, index=index)

    return normalisation_df
