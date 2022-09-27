import numpy as np
from ._transformations import *




def get_estimates_all_inverse(model, images, metadata, normalisation_df):

    predictions = np.array(model.predict([images,metadata[:,-2:]]))
    predictions = np.transpose(predictions,(1,0,2))


    targets_inverse = np.array([targets_zscore__minmax_inv(
            predictions[:,:,i], normalisation_df
        ) for i in range(predictions.shape[-1]) ])

    targets_inverse = np.transpose(targets_inverse,(1,2,0))


    
    
    return targets_inverse

