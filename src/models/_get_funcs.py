import tensorflow as tf


def get_lr_metric(optimizer):
    """
    Show and save learning rate in history.
    """

    @tf.autograph.experimental.do_not_convert
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
