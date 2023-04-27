from sklearn.model_selection import KFold
from datetime import datetime
from .model_arcitectures._model_quantile_simp import quantile_model
import tensorflow as tf


def model_prepare(
    img_size,
    metadata_size,
    model_name,
    batch_size,
    init_lr: float = 0.1,
    Dropout_tcnn: float = 0,
):
    try:
        tf.keras.backend.clear_session()
    except Exception as e:
        print(e)
        pass

    model_bcnn = quantile_model(
        input_size_img=img_size,
        input_size_metadata=metadata_size,
        name=f"{model_name}",
    )
    model_bcnn.make_model(
        Dropout_tcnn=Dropout_tcnn, init_lr=init_lr, batch_size=batch_size
    )

    return model_bcnn


def model_fit(
    model,
    imgs,
    metadata,
    targets,
    callbacks,
    batch_size: int = 20,
    epochs: int = 40,
    verbose: int = 0,
):
    model.fit(
        [imgs, metadata],
        targets,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=verbose,
    )


def model_fit_cv(
    model, image, metadata, targets, folds: int = 10, model_name: str = ""
):

    cvscores = []
    kfold = KFold(n_splits=folds, random_state=None, shuffle=False)
    name_append = datetime.now().strftime("%d_%m_%Y_%H_%M")
    model_name = model_name + name_append

    i = 1
    # cross-validation
    for train, test in kfold.split(image, targets):
        # name of model
        name_append = model_name + f"_f{i}"


