from tensorflow.python.keras.backend import dtype
from window import WindowGenerator
import tensorflow as tf
import pandas as pd
from livelossplot import PlotLossesKeras
from datetime import datetime
import sys
from tensorflow.keras.layers import Reshape, Dense, Dropout, Lambda, LSTM, RNN, SimpleRNN
import tensorflow_probability as tfp
tfpl = tfp.layers
tfb = tfp.bijectors


sys.path.insert(1, '../preprocessing/')
if True:
    from dataset_lib import load_dataset, split_dataset

MAX_EPOCHS = 100


def rmsle(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return tf.keras.backend.sqrt(msle(y_true, y_pred))


def compile_and_fit(model, window, patience=10, max_epochs=MAX_EPOCHS, should_stop=False, lr=0.001, optimizer=tf.optimizers.Adam(lr=0.001), tensorboard=False, lr_schedule_fn=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    if optimizer is None:
        optimizer = tf.optimizers.Adam(lr=lr)

    model.compile(
         run_eagerly=True,
        loss=tf.losses.Huber(),
        optimizer=optimizer,
        metrics=[tf.losses.MeanSquaredLogarithmicError(), tf.losses.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError(), rmsle])

    cbs = [PlotLossesKeras()] + \
        ([early_stopping] if should_stop else []) + \
        ([tf.keras.callbacks.LearningRateScheduler(lr_schedule_fn)]
         if lr_schedule_fn is not None else [])
    if tensorboard:
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        cbs += [tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)]
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=cbs,
                        verbose=2)

    return history


def get_metrics(model, window):
    train_metrics = model.evaluate(window.train)
    val_metrics = model.evaluate(window.val)
    test_metrics = model.evaluate(window.test)

    metrics_names = model.metrics_names
    metrics_names[0] = "huber"
    metrics = pd.DataFrame({
        "names": metrics_names,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    })

    return metrics


def model_generator(model_name, lr, n_features=633, max_epochs=150, relations=[(3, 1), (5, 1), (8, 1), (8, 3), (8, 5), (12, 1), (12, 3), (12, 5), (24, 1), (24, 3), (24, 5)]):
    df = load_dataset("")
    train_df, val_df, test_df = split_dataset(df)

    for r in relations:
        tf.keras.backend.clear_session()
        input_width, OUT_STEPS = r
        window = WindowGenerator(input_width=input_width,
                                 label_width=OUT_STEPS,
                                 shift=OUT_STEPS,
                                 train_df=train_df, val_df=val_df, test_df=test_df)

        if(model_name == "dense"):
            model = get_dense_model(OUT_STEPS, n_features)
        elif(model_name == "simple_rnn"):
            model = get_simple_rnn_model(OUT_STEPS, n_features)
        elif(model_name == "lstm"):
            model = get_simple_lstm_model(OUT_STEPS, n_features)
        elif(model_name == "autoregressive"):
            model = get_feedback(OUT_STEPS, n_features)
        history = compile_and_fit(
            model, window, lr=lr, should_stop=True, max_epochs=max_epochs, tensorboard=True)
        metrics = get_metrics(model, window)
        metrics.to_csv(
            f"../../results/{model_name}/{model_name}-{input_width}-{OUT_STEPS}.csv")


def get_dense_model(OUT_STEPS, n_features):
    if OUT_STEPS == 1:
        return tf.keras.models.Sequential([
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(n_features, activation="relu"),
        ])
    else:
        return tf.keras.models.Sequential([
            Lambda(lambda x: x[:, -1:, :]),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(OUT_STEPS*n_features,
                  kernel_initializer=tf.initializers.zeros),
            Reshape([OUT_STEPS, n_features])
        ])


def get_simple_rnn_model(OUT_STEPS, n_features):
    if OUT_STEPS == 1:
        return tf.keras.Sequential([
            SimpleRNN(100, return_sequences=True),
            Dropout(0.3),
            Dense(128),
            Dropout(0.3),
            Dense(n_features),
        ])
    else:
        return tf.keras.Sequential([
            Lambda(lambda x: x[:, -1:, :]),
            SimpleRNN(100, return_sequences=True),
            Dropout(0.3),
            Dense(128),
            Dropout(0.3),
            Dense(OUT_STEPS*n_features),
            Reshape([OUT_STEPS, n_features])
        ])


def get_simple_lstm_model(OUT_STEPS, n_features):
    if OUT_STEPS == 1:
        return tf.keras.Sequential([
            LSTM(n_features, return_state=False),
            Dropout(0.2),
            Dense(128, activation="relu"),
        ])
    else:
        return tf.keras.Sequential([
            LSTM(n_features, return_sequences=False),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dense(OUT_STEPS*n_features, activation="relu"),
            Reshape([OUT_STEPS, n_features])
        ])


def get_feedback(OUT_STEPS, n_features):
    return FeedBack(units=n_features, steps=OUT_STEPS, n_features=n_features)


class FeedBack(tf.keras.Model):
    def __init__(self, units, steps, n_features):
        super().__init__()
        self.steps = steps
        self.n_features = n_features
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        self.dense = Dense(n_features, activation="relu")

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []

        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.steps):
            # Use the last prediction as input.
            x = prediction
            x_ = tf.concat([x, inputs[:,-1,-3:]], 1)

            # Execute one lstm step.
            x, state = self.lstm_cell(x_, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
