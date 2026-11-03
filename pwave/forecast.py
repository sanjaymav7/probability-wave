"""Turn one window's fit into a forecast distribution for the log-rate H days out."""
import numpy as np
from .fokker_planck import Grid, evolve

GRID = Grid(-0.3, 0.3, 600)


def forecast_quantiles(est, H, levels, grid=GRID):
    s0, g, sr, k = est["sigma0"], est["gamma"], est["s_ref"], est["kappa"]
    x = grid.x
    p = np.exp(-(x - est["d_last"]) ** 2 / (2 * (2 * grid.dx) ** 2))
    p = evolve(p / (p.sum() * grid.dx), grid, k, lambda d: s0 * (1 + g * np.abs(d) / sr), H, 40)
    # If the trend line is off by delta, the forecast is off by (1 - e^{-kH}) delta:
    # the part of the gap to trend that the deviation closes by time H.
    sd = (1 - np.exp(-k * H)) * np.sqrt(est["var_level"])
    if sd > grid.dx / 4:
        kern = np.exp(-0.5 * ((x - x.mean()) / sd) ** 2)
        p = np.convolve(p, kern / kern.sum(), mode="same")
    cdf = np.cumsum(p) * grid.dx
    cdf /= cdf[-1]
    return est["level"] + est["slope"] * H + np.interp(levels, cdf, x)
