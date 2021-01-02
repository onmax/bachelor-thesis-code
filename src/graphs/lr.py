import matplotlib.pyplot as plt


def lr_timeline(history, axis=[1e-6, 1e2, 0, 5], metric="val_mean_squared_logarithmic_error"):
    plt.semilogx(
        history.history["lr"], history.history[metric])
    plt.axis(axis)
