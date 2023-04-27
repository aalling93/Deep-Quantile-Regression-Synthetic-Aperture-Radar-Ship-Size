import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfn = tfp.experimental.nn
class quantile_model_v5:
    def __init__(
        self,
        input_size_img: tuple = (100, 100, 2),
        input_size_metadata: tuple = (2,),
        name: str = "quantile_model_v5",
    ):
        super(quantile_model_v5, self).__init__()

        self.input_size_img = input_size_img
        self.input_size_metadata = input_size_metadata

        self.name = name
        self.model = None
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def data_size(
        self,
        input_size_img: tuple = None,
        input_size_metadata: tuple = None,
        output_size: tuple = None,
    ):

        if input_size_img:
            assert (
                len(input_size_img) == 3
            ), "Input must be of dimension row x column x channels"
            self.input_size_img = input_size_img
        if output_size:
            assert (
                len(output_size) == 1
            ), "Output must be a vector of dimension 1 x features"
            self.output_size = output_size
        if input_size_metadata:
            self.input_size_metadata = input_size_metadata

    def make_model(
        self,
        name: str = "",
        Dropout_tcnn: float = 0.1,
        training: bool = True,
        constrains: list = None,
        init_lr: float = 0.001,
        batch_size: int = 32,
    ):
        """ """
        perc_points = [0.5, 0.25, 0.5, 0.75,0.95]
        if len(name) > 1:
            self.name = name

        inputs_img = tf.keras.layers.Input(self.input_size_img, name="inputs_img")

        inputs_meta = tf.keras.layers.Input(
            self.input_size_metadata, name="inputs_meta"
        )
        alpha = 0.03
        alpha2 = alpha*0.1
        model1 = tf.keras.layers.Conv2D(256, (5, 5), padding="SAME")(inputs_img)
        model1 = tf.keras.activations.elu(model1, alpha)
        model1 = tf.keras.layers.Dropout(Dropout_tcnn)(model1)
        model1 = tf.keras.layers.MaxPool2D((2, 2))(model1)
        model1 = tf.keras.layers.BatchNormalization()(model1)

        model1 = tf.keras.layers.Conv2D(128, (3, 3), padding="SAME")(model1)
        model1 = tf.keras.activations.elu(model1, alpha)
        model1 = tf.keras.layers.Dropout(Dropout_tcnn)(model1)
        model1 = tf.keras.layers.MaxPool2D((2, 2))(model1)
        model1 = tf.keras.layers.BatchNormalization()(model1)

        model1 = tf.keras.layers.Conv2D(28, (3, 3), padding="SAME")(model1)
        model1 = tf.keras.activations.elu(model1, alpha)
        model1 = tf.keras.layers.Dropout(Dropout_tcnn)(model1)
        model1 = tf.keras.layers.MaxPool2D((2, 2))(model1)
        model1 = tf.keras.layers.BatchNormalization()(model1)

        model1 = tf.keras.layers.Flatten()(model1)

        pred1 = tf.keras.layers.Dense(25)(model1)
        pred1 = tf.keras.layers.LeakyReLU(alpha)(pred1)
        pred1 = tf.keras.layers.Dropout(Dropout_tcnn)(pred1)
        pred2 = tf.keras.layers.Dense(25)(model1)
        pred2 = tf.keras.layers.LeakyReLU(alpha)(pred2)
        pred2 = tf.keras.layers.Dropout(Dropout_tcnn)(pred2)
        pred3 = tf.keras.layers.Dense(25)(model1)
        pred3 = tf.keras.layers.LeakyReLU(alpha)(pred3)
        pred3 = tf.keras.layers.Dropout(Dropout_tcnn)(pred3)

        pred1 = tf.keras.layers.Dense(5)(pred1)
        pred1 = tf.keras.layers.LeakyReLU(alpha)(pred1)
        pred2 = tf.keras.layers.Dense(5)(pred2)
        pred2 = tf.keras.layers.LeakyReLU(alpha)(pred2)
        pred3 = tf.keras.layers.Dense(5)(pred3)
        pred3 = tf.keras.layers.LeakyReLU(alpha)(pred3)


        # Relu is used at the end to force the output to be positive (which we know it is).
        # LeakyRelu is used throughout the other layers to avoid the dying relu problem (whis is bad)
        # Whenever you get the same predictions for all input. It could be the dying rely problems.
        pred1 = tf.keras.layers.Dense(5,activation = 'relu',name='length')(pred1)
        #pred1 = tf.keras.layers.LeakyReLU(alpha2)(pred1)
        pred2 = tf.keras.layers.Dense(5,activation = 'relu',name='width')(pred2)
        #pred2 = tf.keras.layers.LeakyReLU(alpha2)(pred2)
        pred3 = tf.keras.layers.Dense(5,activation = 'relu',name='ratio')(pred3)
        #pred3 = tf.keras.layers.LeakyReLU(alpha2)(pred3)

        self.model = tf.keras.Model(
            inputs=[inputs_img, inputs_meta],
            outputs=[pred1, pred2, pred1],
            name=self.name,
        )
        optimizer = tf.optimizers.Adam(learning_rate=init_lr, clipnorm=1, clipvalue=1.0)
        lr_metric = get_lr_metric(optimizer)
        
        
        self.model.compile(
            loss=[
                QuantileLoss(perc_points),
                QuantileLoss(perc_points),
                QuantileLoss(perc_points),
            ],
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.MeanAbsolutePercentageError(),
                lr_metric
            ],
        )






