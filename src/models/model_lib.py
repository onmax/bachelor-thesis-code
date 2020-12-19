import tensorflow as tf
from livelossplot import PlotLossesKeras
from datetime import datetime
import os
import inspect
import sys


os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))) + "../graphs/"
sys.path.insert(1, os.path.join(sys.path[0], '../graphs'))
if True:
    from predictions import plot_predictions

MAX_EPOCHS = 100


def compile_and_fit(model, window, patience=10, max_epochs=MAX_EPOCHS, should_stop=False, lr=0.001, optimizer=tf.optimizers.Adam(lr=0.001), tensorboard=False):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    if optimizer is None:
        optimizer = tf.optimizers.Adam(lr=lr)

    model.compile(
        # loss=tf.losses.MeanSquaredError(),
        loss=tf.losses.MeanSquaredLogarithmicError(),
        optimizer=optimizer,
        metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()])

    cbs = [PlotLossesKeras()] + ([early_stopping] if should_stop else [])
    if tensorboard:
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        cbs += [tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)]
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=cbs,
                        verbose=2)

    plot_predictions(window, model, 10)
    return history
