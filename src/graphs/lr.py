import matplotlib.pyplot as plt


def lr_timeline(history, axis=[1e-6, 1e2, 0, 5]):
    plt.semilogx(
        history.history["lr"], history.history["val_mean_squared_logarithmic_error"])
    plt.axis(axis)
