"""Price as a probability density, moved forward by the Fokker-Planck equation.

For a deviation d from trend with dynamics  dd = -kappa d dt + sigma(d) dW,
the density obeys

    dp/dt = d/dd [ kappa d p ]  +  1/2 d^2/dd^2 [ sigma(d)^2 p ].

Discretised in flux (finite-volume) form so probability can only move between
neighbouring cells, never appear or vanish, and stepped implicitly so the
step size is set by accuracy rather than stability.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded


@dataclass
class Grid:
    lo: float
    hi: float
    n: int = 400

    @property
    def dx(self):
        return (self.hi - self.lo) / self.n

    @property
    def x(self):
        return self.lo + (np.arange(self.n) + 0.5) * self.dx


def _operator(grid, kappa, sigma_fn):
    """Tridiagonal L with dp/dt = L p, zero-flux walls at both ends."""
    x, dx, n = grid.x, grid.dx, grid.n
    xf = grid.lo + np.arange(1, n) * dx            # interior faces
    a = -kappa * xf                                # drift velocity at faces
    D = 0.5 * sigma_fn(x) ** 2                     # diffusion coefficient at cell centres
    # Flux across face i+1/2:  F = a (p_i + p_{i+1})/2 - (D_{i+1} p_{i+1} - D_i p_i)/dx.
    # Central drift is second order; the cell Peclet number |a| dx / D stays far
    # below 2 on these grids, so it doesn't oscillate.
    cl = 0.5 * a + D[:-1] / dx                     # coefficient of p_i in F_{i+1/2}
    cr = 0.5 * a - D[1:] / dx                      # coefficient of p_{i+1}
    main = np.zeros(n); upper = np.zeros(n); lower = np.zeros(n)
    # dp_i/dt = (F_{i-1/2} - F_{i+1/2}) / dx
    main[:-1] -= cl / dx; upper[1:] -= cr / dx     # outgoing through right face
    main[1:] += cr / dx; lower[:-1] += cl / dx     # incoming through left face
    return lower, main, upper


def evolve(p0, grid, kappa, sigma_fn, T, steps=80):
    """Crank-Nicolson for T time units. Returns the density at T (integrates to 1)."""
    lower, main, upper = _operator(grid, kappa, sigma_fn)
    h = T / steps
    ab = np.zeros((3, grid.n))
    ab[0, 1:] = -0.5 * h * upper[1:]
    ab[1] = 1 - 0.5 * h * main
    ab[2, :-1] = -0.5 * h * lower[:-1]
    p = p0.copy()
    for _ in range(steps):
        rhs = p + 0.5 * h * (main * p)
        rhs[:-1] += 0.5 * h * upper[1:] * p[1:]
        rhs[1:] += 0.5 * h * lower[:-1] * p[:-1]
        p = solve_banded((1, 1), ab, rhs)
    return p


def ou_closed_form(x, d0, kappa, sigma, t):
    m = d0 * np.exp(-kappa * t)
    v = sigma**2 * (1 - np.exp(-2 * kappa * t)) / (2 * kappa)
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
