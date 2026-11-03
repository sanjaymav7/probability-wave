"""The solver against the one case with a closed form, and the properties it must never break."""
import numpy as np
from pwave import Grid, evolve, ou_closed_form, crps_gaussian, crps_quantiles
from scipy.stats import norm

K, S, D0, T = 0.03, 0.0022, 0.03, 20


def _err(n):
    g = Grid(-0.2, 0.2, n)
    p0 = ou_closed_form(g.x, D0, K, S, 0.5)
    p = evolve(p0, g, K, lambda x: np.full_like(x, S), T, 80)
    exact = ou_closed_form(g.x, D0, K, S, T + 0.5)
    return p, g, np.abs(p - exact).max() / exact.max()


def test_matches_ou_transition_density():
    _, _, err = _err(800)
    assert err < 1e-3


def test_second_order_in_space():
    e1, e2 = _err(200)[2], _err(400)[2]
    assert 3.5 < e1 / e2 < 4.5          # halving dx quarters the error


def test_probability_is_conserved():
    p, g, _ = _err(400)
    assert abs(p.sum() * g.dx - 1) < 1e-12
    assert p.min() > -1e-12


def test_quantile_crps_approaches_exact_gaussian_crps():
    levels = np.linspace(0.001, 0.999, 999)
    q = norm.ppf(levels, 0.1, 0.5)
    assert abs(crps_quantiles(q, levels, 0.7) - crps_gaussian(0.1, 0.5, 0.7)) < 5e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
