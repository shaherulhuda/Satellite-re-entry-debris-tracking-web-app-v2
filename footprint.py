"""Where on Earth an object can pass over, from its inclination alone."""
from __future__ import annotations

import numpy as np


def max_latitude(inclination_deg: float) -> float:
    """Highest latitude reached (retrograde orbits count as 180 - i)."""
    i = float(inclination_deg) % 360.0
    i = 360.0 - i if i > 180 else i
    return min(i, 180.0 - i)


def surface_fraction(inclination_deg: float) -> float:
    """Share of Earth's surface area lying within +/- max latitude: sin(phi_max)."""
    return float(np.sin(np.radians(max_latitude(inclination_deg))))


def latitude_distribution(inclination_deg: float, bin_deg: float = 2.0):
    """Share of time a circular orbit spends in each latitude bin.

    sin(lat) = sin(i) sin(u) with u uniform, so the CDF is F(phi) = 1/2 + asin(sin(phi)/sin(i))/pi.
    Returns (bin_centres_deg, probability_per_bin); probabilities sum to 1.
    """
    phi_max = max_latitude(inclination_deg)
    s = np.sin(np.radians(phi_max))
    edges = np.arange(-90.0, 90.0 + bin_deg, bin_deg)
    if s <= 0:
        p = np.zeros(len(edges) - 1)
        p[np.argmin(np.abs(0.5 * (edges[:-1] + edges[1:])))] = 1.0
        return 0.5 * (edges[:-1] + edges[1:]), p
    x = np.clip(np.sin(np.radians(edges)) / s, -1.0, 1.0)
    cdf = 0.5 + np.arcsin(x) / np.pi
    return 0.5 * (edges[:-1] + edges[1:]), np.diff(cdf)
