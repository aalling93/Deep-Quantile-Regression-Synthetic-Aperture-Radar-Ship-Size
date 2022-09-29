from ._callbacks import *
from ._util import *
from .model_arcitectures._model_quantile_simp import quantile_model
from .model_arcitectures._model_quantule_v2 import quantile_model as qmodel_v2
from .model_arcitectures._fcn_quantile import fcn_quantile


class Model:
    """'"""

    def __init__(self, seed: int = 42, gpu: int = 1, memory: int = 250000):
        super(Model, self).__init__()
        # setting Seed
        seed_everything(42)

        # init GPU.
        if gpu != -1:
            load_gpu(which=gpu, memory=memory)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __str__(self):
        a = "i dont know yet."

    def data_load(self, images, metadata, target):

        self.train_images = images
        self.train_metadata = metadata
        self.train_target = target

    def model_load(
        self,
        which_model: str = "quantile_model",
        model_name: str = "quantile_v3",
        init_lr: float = 0.0001,
        batch_size: int = 24,
        cycle_length: int = 100,
        cycle_mult_factor: float = 2,
        validation_split: float = 0.2,
        print_callback: bool = True,
        early_stopping: int = 500,
    ):
        """ """
        self.which_model = which_model
        clean_session()
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.steps_pr_epoch = int(
            np.ceil(
                (self.train_images.shape[0] * (1 - self.validation_split))
                / self.batch_size
            )
        )

        if which_model.lower() == "quantile_model":
            self.model = quantile_model(
                input_size_img=self.train_images[0].shape,
                input_size_metadata=self.train_metadata[0, -2:].shape,
                name=f"{model_name}",
            )
            self.model.make_model(
                Dropout_tcnn=0.10, init_lr=init_lr, batch_size=batch_size
            )
        
        if which_model.lower() == "quantile_model_v2":
            self.model = qmodel_v2(
                input_size_img=self.train_images[0].shape,
                input_size_metadata=self.train_metadata[0, -2:].shape,
                name=f"{model_name}",
            )
            self.model.make_model(
                Dropout_tcnn=0.10, init_lr=init_lr, batch_size=batch_size
            )

        if which_model.lower() == "fcn_quantile":
            print('quantile')
            self.model = fcn_quantile(
                input_size_img=(None,None,self.train_images[0].shape[-1]),
                input_size_metadata=self.train_metadata[0, -2:].shape,
                name=f"{model_name}",
            )
            self.model.make_model(
                Dropout_tcnn=0.10, init_lr=init_lr, batch_size=batch_size
            )



        self.callbacks = get_callbacks(
            self.model,
            optimizer_method="cosine",
            steps_per_epoch=self.steps_pr_epoch,
            wd_norm=0.004,
            eta_min=0.00000000002,
            eta_max=1,
            eta_decay=0.005,
            cycle_length=cycle_length,
            cycle_mult_factor=cycle_mult_factor,
            print_callback=print_callback,
            early_stopping=early_stopping,
        )

    def model_train(self, epochs: int = 50, verbose: int = 1):
        self.epochs = epochs
        

        if self.which_model.lower() == "fcn_quantile":
            training_history = self.model.model.fit(
            [self.train_images, self.train_metadata[:, -2:]],
            [self.train_target[0], self.train_target[1], self.train_target[2]],
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=verbose)
        


        else:
            training_history = self.model.model.fit(
                [self.train_images, self.train_metadata[:, -2:]],
                [self.train_target[:, 0], self.train_target[:, 1], self.train_target[:, 2]],
                batch_size=self.batch_size,
                epochs=epochs,
                validation_split=self.validation_split,
                verbose=verbose)
        self.training_history = training_history

    def model_train_v2(self, epochs: int = 50, verbose: int = 1):
        self.epochs = epochs
        training_history = self.model.model.fit(
            [self.train_images_vv,self.train_images_vh, self.train_metadata[:, -2:]],
            [self.train_target[:, 0], self.train_target[:, 1], self.train_target[:, 2]],
            batch_size=self.batch_size,
            epochs=epochs,
            validation_split=self.validation_split,
            verbose=verbose,
        )
        self.training_history = training_history
