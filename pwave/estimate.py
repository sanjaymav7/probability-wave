"""Fit trend, mean reversion and a local-vol curve from one window of history."""
import numpy as np


def fit_window(x_past, window=180):
    """Fit on everything observed so far, without looking ahead.

    The long-run trend comes from the full past (a short window would soak up
    the very deviations the model is meant to revert); mean reversion and the
    local-vol curve come from the last `window` days around that trend.
    """
    n_all = len(x_past)
    t_all = np.arange(n_all)
    b, a = np.polyfit(t_all, x_past, 1)
    x = x_past[-window:]
    n = len(x)
    t = t_all[-window:]
    d = x - (a + b * t)
    y, z = d[1:], d[:-1]
    phi = np.dot(z, y) / np.dot(z, z)
    # Least-squares AR(1) is biased toward faster reversion in short samples;
    # Kendall (1954): E[phi_hat] - phi ~ -(1 + 3 phi) / n.
    phi = min(phi + (1 + 3 * phi) / len(y), 0.999)
    kappa = np.clip(1 - phi, 1e-3, 0.5)
    e = y - phi * z
    var_kappa = e.var() / np.dot(z, z)
    s_ref = np.sqrt(np.mean(d**2))
    # E|e| = sqrt(2/pi) * sigma0 * (1 + gamma |d|/s_ref): a straight line in |d|/s_ref.
    u = np.abs(z) / s_ref
    X = np.column_stack([np.ones_like(u), u])
    coef, res, *_ = np.linalg.lstsq(X, np.abs(e) * np.sqrt(np.pi / 2), rcond=None)
    sigma0 = max(coef[0], 1e-5)
    gamma = max(coef[1] / sigma0, 0.0)
    # |e| pins down the curve's shape; its level is biased low when vol moves
    # around inside the window (E|e| ~ mean sigma, not rms sigma). Rescale so
    # the model's rms vol matches the residuals'.
    sigma0 *= np.sqrt(np.mean(e**2) / np.mean((sigma0 * (1 + gamma * u)) ** 2))
    resid = np.abs(e) * np.sqrt(np.pi / 2) - X @ coef
    cov = resid.var() * np.linalg.inv(X.T @ X)
    var_gamma = cov[1, 1] / sigma0**2
    # Variance of the fitted trend's endpoint. Deviations are autocorrelated, so
    # the history holds only n (1 - phi) / (1 + phi) independent looks at d, and
    # the endpoint of an OLS line has about 4/n times the noise variance.
    n_eff = max(n_all * (1 - phi) / (1 + phi), 2.0)
    var_level = 4 * np.mean(d**2) / n_eff
    return dict(slope=b, level=a + b * (n_all - 1), var_level=var_level, d_last=d[-1], kappa=kappa, sigma0=sigma0,
                gamma=gamma, s_ref=s_ref, var_kappa=var_kappa, var_gamma=var_gamma)


def shrink(est, history):
    """Empirical-Bayes: pull noisy per-window kappa and gamma toward the cross-window mean.

    Weight on the window's own estimate = tau^2 / (tau^2 + v), where v is that
    estimate's sampling variance and tau^2 the spread of the true parameter
    across windows (total variance minus mean sampling variance, floored at 0).
    """
    out = dict(est)
    if len(history) < 20:
        return out
    for key in ("kappa", "gamma"):
        vals = np.array([h[key] for h in history])
        noise = np.array([h["var_" + key] for h in history])
        tau2 = max(vals.var() - noise.mean(), 0.0)
        w = tau2 / (tau2 + est["var_" + key]) if tau2 + est["var_" + key] > 0 else 0.0
        out[key] = w * est[key] + (1 - w) * vals.mean()
    return out
