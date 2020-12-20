import tensorflow as tf
from livelossplot import PlotLossesKeras
from datetime import datetime
import sys


sys.path.insert(1, '../preprocessing/')
if True:
    from dataset_lib import load_dataset, split_dataset

MAX_EPOCHS = 100


def compile_and_fit(model, window, patience=10, max_epochs=MAX_EPOCHS, should_stop=False, lr=0.001, optimizer=tf.optimizers.Adam(lr=0.001), tensorboard=False, with_lr_schedule=False):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    if with_lr_schedule:
        lr = 1e-6

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-6 * 10**(epoch/10))
    if optimizer is None:
        optimizer = tf.optimizers.Adam(lr=lr)

    model.compile(
        loss=tf.losses.Huber(),
        optimizer=optimizer,
        metrics=[tf.losses.MeanSquaredLogarithmicError(), tf.losses.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()])

    cbs = [PlotLossesKeras()] + \
        ([early_stopping] if should_stop else []) + \
        ([lr_schedule] if with_lr_schedule else [])
    if tensorboard:
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        cbs += [tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)]
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=cbs,
                        verbose=2)

    return history


def get_datasets():
    df = load_dataset()
    train_df, val_df = split_dataset(df, train_from=datetime(2018, 1, 1))
    print(f"Training on {train_df.columns.to_list()}")
    print(f"Validating on {val_df.columns.to_list()}")
    return train_df, val_df
