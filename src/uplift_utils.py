"""Shared helper functions for uplift evaluation."""
import numpy as np


def qini_curve(y, t, score):
    """Extra purchases caused by the email when targeting customers
    in order of highest score first."""
    order = np.argsort(-score)
    y = np.asarray(y)[order]
    t = np.asarray(t)[order]
    n = len(y)

    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)

    ratio = np.divide(n_t, n_c, out=np.zeros(n, dtype=float), where=n_c > 0)
    qini = y_t - y_c * ratio

    x = np.insert(np.arange(1, n + 1) / n, 0, 0.0)
    qini = np.insert(qini, 0, 0.0)
    return x, qini


def area(x, y):
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def qini_coefficient(y, t, score):
    """Area between the model curve and the random-targeting line.
    0 = no better than random. Higher is better."""
    x, q = qini_curve(y, t, score)
    return area(x, q) - area(x, x * q[-1])


def extra_buyers_at(y, t, score, frac):
    """(extra buyers for the model, extra buyers for random) at top `frac`."""
    x, q = qini_curve(y, t, score)
    i = int(frac * (len(x) - 1))
    return float(q[i]), float(q[-1] * x[i])