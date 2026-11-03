"""Continuous ranked probability score: lower is better, and it scores the whole distribution."""
import numpy as np
from scipy.stats import norm


def crps_from_cdf(grid_x, cdf, y):
    step = (grid_x >= y).astype(float)
    return np.trapz((cdf - step) ** 2, grid_x)


def crps_gaussian(mu, sigma, y):
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_quantiles(qs, levels, y):
    """CRPS = 2 * integral of pinball loss over levels, approximated on the given levels."""
    qs, levels = np.asarray(qs), np.asarray(levels)
    pin = np.where(y >= qs, levels * (y - qs), (1 - levels) * (qs - y))
    return 2 * np.trapz(pin, levels) / (levels[-1] - levels[0])
