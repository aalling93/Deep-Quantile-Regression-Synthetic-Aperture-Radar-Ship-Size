import glob
from ._estimation import *
from ._evaluate import *
from ._get_funcs import *
from ._transformations import *
from ._load import *
from .visualize import *


class Analyse_model:
    """'"""

    def __init__(self, model):
        super(Analyse_model, self).__init__()

        try:
            self.model = self.model.model
        except:
            try:
                self.model = model
            except Exception as e:
                print("Error in loading model")

        self.img_size, self.metadata_size, self.layer_names = model_parms(model)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __str__(self):
        a = "i dont know yet."

    def get_history(self):
        self.history = load_history(self.model)

    def get_data(self, list: list = []):
        """list =  [images,metadata,targets]"""
        self.images, self.metadata, self.targets = load_data(list)

    def get_scaling(self, list: list = []):
        """list =  [(
            minimum_values,
            metadata_training_minimum_values,
            minimum_values_targets,
            maximum_values,
            metadata_training_maximum_values,
            maximum_values_targets,
            mean_values,
            metadata_training_mean_values,
            mean_values_targets,
            std_values,
            metadata_training_std_values,
            std_values_targets,
        )]"""
        self.normalisation_df = load_scaling(list)

    def inverse_values(self):
        """Getting"""
        (
            self.images_inverse,
            self.metadata_inverse,
            self.targets_inverse,
            self.true_lengths,
            self.true_widths,
            self.true_ratios,
        ) = all_inverse_values(
            self.images, self.metadata, self.targets, self.normalisation_df
        )
        self.cog = self.metadata_inverse[:, -2]
        self.sog = self.metadata_inverse[:, -1]
        self.mmsi = self.metadata_inverse[:, 0].astype(int)
        

    

    def all_estimates_inverse(self):
        self.targets_inverse = get_estimates_all_inverse(
            self.model,
            self.images,
            self.metadata,
            self.normalisation_df
            )
        self.predicted_length_inv = self.targets_inverse[:, 0].astype(int)
        self.predicted_width_inv = self.targets_inverse[:, 1].astype(int)
        self.predicted_ratio_inv = self.targets_inverse[:, 2]

        self.predicted_length_inv_q01 = self.targets_inverse[:, 0,0].astype(int)
        self.predicted_width_inv_q01 = self.targets_inverse[:, 1,0].astype(int)
        self.predicted_ratio_inv_q01 = self.targets_inverse[:, 2,0].astype(int)

        self.predicted_length_inv_q25 = self.targets_inverse[:, 0,1].astype(int)
        self.predicted_width_inv_q25 = self.targets_inverse[:, 1,1].astype(int)
        self.predicted_ratio_inv_q25 = self.targets_inverse[:, 2,1].astype(int)

        self.predicted_length_inv_q50 = self.targets_inverse[:, 0,2].astype(int)
        self.predicted_width_inv_q50 = self.targets_inverse[:, 1,2].astype(int)
        self.predicted_ratio_inv_q50 = self.targets_inverse[:, 2,2].astype(int)

        self.predicted_length_inv_q75 = self.targets_inverse[:, 0,3].astype(int)
        self.predicted_width_inv_q75 = self.targets_inverse[:, 1,3].astype(int)
        self.predicted_ratio_inv_q75 = self.targets_inverse[:, 2,3].astype(int)

        self.predicted_length_inv_q99 = self.targets_inverse[:, 0,4].astype(int)
        self.predicted_width_inv_q99 = self.targets_inverse[:, 1,4].astype(int)
        self.predicted_ratio_inv_q99 = self.targets_inverse[:, 2,4].astype(int)

    def accuracy(self):
        self.length_acc,self.width_acc = get_accuracy(self.true_lengths, self.predicted_length_inv, self.true_widths, self.predicted_width_inv)
         
    def confidence_intervals(self):
        self.length_99CI = self.predicted_length_inv[:,-1]-self.predicted_length_inv[:,0]
        self.length_75CI = self.predicted_length_inv[:,-2]-self.predicted_length_inv[:,1]

        self.width_99CI = self.predicted_width_inv[:,-1]-self.predicted_width_inv[:,0]
        self.width_75CI = self.predicted_width_inv[:,-2]-self.predicted_width_inv[:,1]


    def abs_errors(self):
        self.length_errors_q50,self.width_errors_q50,self.normalized_erros =  get_abs_errors(self.true_lengths, self.predicted_length_inv_q50, self.true_widths, self.predicted_width_inv_q50)
        self.length_errors_q25,self.width_errors_q25,__ =  get_abs_errors(self.true_lengths, self.predicted_length_inv_q25, self.true_widths, self.predicted_width_inv_q25)
        self.length_errors_q75,self.width_errors_q75,__ =  get_abs_errors(self.true_lengths, self.predicted_length_inv_q75, self.true_widths, self.predicted_width_inv_q75)


    def mse_errors(self):

        self.errors_q50 =  get_mse_errors(self.errors_q50, self.predicted_length_inv_q50, self.true_widths, self.predicted_width_inv_q50)
        self.errors_q25 =  get_mse_errors(self.true_lengths, self.predicted_length_inv_q25, self.true_widths, self.predicted_width_inv_q25)
        self.errors_q75 =  get_mse_errors(self.true_lengths, self.predicted_length_inv_q75, self.true_widths, self.predicted_width_inv_q75)
        self.errors_q01 =  get_mse_errors(self.true_lengths, self.predicted_length_inv_q01, self.true_widths, self.predicted_width_inv_q01)
        self.errors_q99 =  get_mse_errors(self.true_lengths, self.predicted_length_inv_q99, self.true_widths, self.predicted_width_inv_q99)

