import glob
import pandas as pd




def model_parms(model):
    layer_names = [layer.name for layer in model.layers]
    for i in range(len(model.input)):
        if i == 0:
            img_size = model.input[0].shape[1:]
        if i == 1:
            metadata_size = model.input[1].shape[1:]
    return img_size, metadata_size, layer_names