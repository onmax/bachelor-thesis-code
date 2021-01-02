import matplotlib.pyplot as plt
import random
import pandas as pd


def plot_predictions(window, model=None, max_subplots=5):
    inputs, labels = window.example
    plot_col = random.choice(window.label_columns)
    plot_col_index = window.column_indices[plot_col]

    plt.figure(figsize=(12, 8))
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(plot_col)

        # Plot training data
        plt.plot(window.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        label_col_index = window.label_columns_indices.get(
            plot_col, None)

        # Plot labels if they exists
        if label_col_index is None:
            continue
        plt.scatter(window.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)

        # Plot predictions if they exists
        if model is not None:
            predictions = model(inputs)
            plt.scatter(window.label_indices, predictions[n, :, label_col_index][plot_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)
    plt.legend()
    plt.xlabel('Time [h]')
    plt.show()


def convert_to_big_station(df):
    quantity_columns = df.columns.difference(["hour", "day_of_week", "month"])
    df['quantity'] = df[quantity_columns].sum(axis=1)
    df = df.resample(pd.Timedelta('1 days')).sum()
    return df


def plot_stations(df):
    df = convert_to_big_station(df)
    plt.plot(df.index, df["quantity"])

def plot_predictions(y, y_hat):
    big_y = convert_to_big_station(y)
    big_y_hat = convert_to_big_station(y_hat)
    plt.plot(big_y.index, big_y["quantity"])
    plt.plot(big_y_hat.index, big_y_hat["quantity"])
