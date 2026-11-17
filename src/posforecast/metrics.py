import numpy as np


def mape(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    m = (a != 0) & ~np.isnan(a) & ~np.isnan(f)
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100) if m.any() else float("nan")


def wape(actual, forecast) -> float:
    """Total absolute error / total actual. Less sensitive to tiny series than MAPE."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    m = ~np.isnan(a) & ~np.isnan(f)
    return float(np.abs(a[m] - f[m]).sum() / np.abs(a[m]).sum() * 100) if m.any() else float("nan")
