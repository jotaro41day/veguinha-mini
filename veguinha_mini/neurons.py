"""Parâmetros Izhikevich, com heterogeneidade célula a célula."""

import numpy as np


PARAMETERS = {
    "RS": (0.02, 0.20, -65.0, 8.0),
    "IB": (0.02, 0.20, -55.0, 4.0),
    "CH": (0.02, 0.20, -50.0, 2.0),
    "FS": (0.10, 0.20, -65.0, 2.0),
    "LTS": (0.02, 0.25, -65.0, 2.0),
}


def make_parameters(inhibitory: np.ndarray, rng: np.random.Generator):
    n = len(inhibitory)
    kinds = np.empty(n, dtype="<U3")
    kinds[~inhibitory] = rng.choice(("RS", "IB", "CH"), size=(~inhibitory).sum(), p=(0.75, 0.15, 0.10))
    kinds[inhibitory] = rng.choice(("FS", "LTS"), size=inhibitory.sum(), p=(0.7, 0.3))
    values = np.array([PARAMETERS[kind] for kind in kinds], dtype=float)
    values *= rng.normal(1.0, (0.08, 0.025, 0.015, 0.08), size=(n, 4))
    values[:, 0] = np.maximum(values[:, 0], 0.005)
    values[:, 1] = np.maximum(values[:, 1], 0.05)
    values[:, 3] = np.maximum(values[:, 3], 0.1)
    return kinds, values
