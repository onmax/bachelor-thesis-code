import matplotlib.pyplot as plt
import random


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
