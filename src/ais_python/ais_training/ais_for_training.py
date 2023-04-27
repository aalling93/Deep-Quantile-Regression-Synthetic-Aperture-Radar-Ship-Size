from ._util import *
from .normalize import *

class Ais_for_training():
    def __init__(self,df):
        self.df = df


        self.stats = self.df.describe()

        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def get_data(self):
        
        self.df = resample_all_ids(self.df)
        res = scale_zscore_minmax(self.df)
        self.scaled_df = res.df
        self.zscore_stats = res.zscore_stats
        self.minmax_stats = res.minmax_stats

        samples,targets,features= get_datasets(self.scaled_df)

        self.samples = samples
        self.targets = targets
        self.features = features










    
  