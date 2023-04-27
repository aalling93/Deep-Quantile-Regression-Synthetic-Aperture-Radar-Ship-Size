import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K


def get_callbacks(
    Model,
    optimizer_method: str = "cosine",
    steps_per_epoch=10,
    wd_norm=0.004,
    eta_min=0.00000000002,
    eta_max=1,
    eta_decay=0.005,
    cycle_length=100,
    cycle_mult_factor=2,print_callback:bool=True,early_stopping:int=500
):
    """ """

    os.makedirs(f"models", exist_ok=True)
    os.makedirs(f"models/{Model.model.name}", exist_ok=True)

    

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=early_stopping,
        min_delta=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    
    cbk = Saving_callback(Model)
    sace_cbk = Saving_callback(Model)
    cb_wrwd = WRWDScheduler(
            steps_per_epoch=steps_per_epoch,
            lr=K.eval(Model.model.optimizer.lr),
            wd_norm=wd_norm,
            eta_min=eta_min,
            eta_max=eta_max,
            eta_decay=eta_decay,
            cycle_length=cycle_length,
            cycle_mult_factor=cycle_mult_factor,
        )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.8, patience=5, min_lr=1e-15
        )

    callbacks = [early_stop,cbk]
    if print_callback==True:
        callbacks.append(sace_cbk)
    if optimizer_method.lower()=="cosine":
        callbacks.append(cb_wrwd)
    else:
        callbacks.append(reduce_lr)

    return callbacks

    


class Saving_callback(tf.keras.callbacks.Callback):
    def __init__(self, Model):
        super().__init__()
        self.Model = Model
        self.epoch_count = 0
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            try:
                if self.Model.model.history.history["val_loss"][-1] == min(
                    self.Model.model.history.history["val_loss"]
                ):
                
                    self.Model.model.save(f"models/{self.model.name}/model_{self.model.name}_lowest_loss.h5",
                        overwrite=True,
                    )
                    pd.DataFrame(self.Model.model.history.history).to_pickle(
                        f"models/{self.Model.model.name}/history_lowest_loss_epoch_{self.Model.model.name}.pkl"
                    )
            except:
                if self.Model.model.history.history["loss"][-1] == min(
                    self.Model.model.history.history["loss"]
                ):
                    self.Model.model.save(f"models/{self.model.name}/model_{self.model.name}_lowest_loss.h5",
                        overwrite=True,
                    )
                    pd.DataFrame(self.Model.model.history.history).to_pickle(
                        f"models/{self.Model.model.name}/history_lowest_loss_epoch_{self.Model.model.name}.pkl"
                    )

            lr = K.get_value(self.Model.model.optimizer.lr)
            self.learning_rates.append(lr)
            self.Model.model.save_weights(
                f"models/{self.Model.model.name}/weights_{self.Model.model.name}_newest_epoch.h5",
                overwrite=True,
            )
            pd.DataFrame(self.Model.model.history.history).to_pickle(
                f"models/{self.Model.model.name}/history_newest_epoch_{self.Model.model.name}.pkl"
            )

        if epoch in [0, 1, 2, 10, 20, 50, 70, 100, 200, 300, 500,800,1000,1400,2000,3000,4000,5000,6000,7000,9000,10000,12000,15000]:
            self.Model.model.save_weights(
                f"models/{self.Model.model.name}/weights_{self.Model.model.name}_epoch_{epoch}.h5"
            )


class Printing_callback(tf.keras.callbacks.Callback):
    def __init__(self, Model):
        super().__init__()
        self.Model = Model
        self.epoch_count = 0
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch%10==0:
            print(f'\n====================================\nEpoch: {epoch}/{self.Model.epochs}  \nLoss: {self.Model.model.history.history["loss"][-1]} \n Val_loss: {self.Model.model.history.history["val_loss"][-1]}\n')
            




class WRWDScheduler(tf.keras.callbacks.Callback):
    

    """ """

    @tf.autograph.experimental.do_not_convert
    def __init__(
        self,
        steps_per_epoch,
        lr,
        wd_norm=0.004,
        eta_min=0.000002,
        eta_max=1,
        eta_decay=0.01,
        cycle_length=10,
        cycle_mult_factor=2.5,
    ):
        """Constructor for warmup learning rate scheduler"""

        super(WRWDScheduler, self).__init__()
        self.lr = lr
        self.wd_norm = wd_norm

        self.steps_per_epoch = steps_per_epoch

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_decay = eta_decay

        self.steps_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.cycle_mult_factor = cycle_mult_factor

        self.wd = wd_norm / (steps_per_epoch * cycle_length) ** 0.5

        self.history = defaultdict(list)
        self.batch_count = 0
        self.learning_rates = []

        self.batch_count = 0
        self.learning_rates = []

    @tf.autograph.experimental.do_not_convert
    def cal_eta(self):
        """Calculate eta"""
        fraction_to_restart = self.steps_since_restart / (
            self.steps_per_epoch * self.cycle_length
        )
        eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1.0 + np.cos(fraction_to_restart * np.pi)
        )
        return eta

    @tf.autograph.experimental.do_not_convert
    def on_train_batch_begin(self, batch, logs={}):
        """update learning rate and weight decay"""
        eta = self.cal_eta()
        # self.model.optimizer._learning_rate = eta * self.lr
        lr = eta * self.lr

        K.set_value(self.model.optimizer.lr, lr)
        # self.model.optimizer._weight_decay = eta * self.wd

    @tf.autograph.experimental.do_not_convert
    def on_train_batch_end(self, batch, logs={}):
        """Record previous batch statistics"""
        logs = logs or {}

        # self.history['wd'].append(self.model.optimizer.optimizer._weight_decay)
        for k, v in logs.items():
            self.history[k].append(v)

        self.steps_since_restart += 1

    @tf.autograph.experimental.do_not_convert
    def on_epoch_end(self, epoch, logs={}):

        """Check for end of current cycle, apply restarts when necessary"""

        def on_epoch_end(self, epoch, logs=None):
            print(K.eval(self.model.optimizer.lr))

        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
        # self.model.history.history.append(lr)

        self.history["lr"].append(lr)
        self.learning_rates.append(lr)

        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.cycle_mult_factor)
            self.next_restart += self.cycle_length
            self.eta_min *= self.eta_decay
            self.eta_max *= self.eta_decay
            self.wd = self.wd_norm / (self.steps_per_epoch * self.cycle_length) ** 0.5
