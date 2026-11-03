"""Synthetic USD/INR with a known answer.

log-rate = slow trend + mean-reverting deviation whose volatility rises with
the size of the deviation (a local-volatility smile), under vol regimes that
shift a few times a year. Because every parameter is planted, a forecaster
that does well here is reading real structure, not luck in one sample.
"""
import numpy as np


def simulate_usdinr(years=6, seed=7, days_per_year=252):
    rng = np.random.default_rng(seed)
    n = years * days_per_year
    t = np.arange(n)
    trend = np.log(74.0) + 0.028 / days_per_year * t          # ~2.8%/yr depreciation of INR
    regime = np.repeat(rng.choice([0.6, 1.0, 1.7], size=n // 63 + 1), 63)[:n]
    sigma0, gamma, kappa = 0.0022, 0.35, 0.03
    s_ref = sigma0 / np.sqrt(2 * kappa)                        # stationary scale of d
    d = np.zeros(n)
    for i in range(1, n):
        vol = sigma0 * regime[i] * (1 + gamma * abs(d[i - 1]) / s_ref)
        d[i] = d[i - 1] - kappa * d[i - 1] + vol * rng.standard_normal()
    return dict(x=trend + d, trend=trend, regime=regime,
                truth=dict(kappa=kappa, sigma0=sigma0, gamma=gamma))
