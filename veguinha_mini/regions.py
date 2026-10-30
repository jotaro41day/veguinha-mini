"""Populações e graus de entrada por alvo da rede isolada."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    cells: int
    inhibitory: int
    bias: float


@dataclass(frozen=True)
class Pathway:
    source: str
    target: str
    in_degree: int
    weight: float


BASE_REGIONS = (
    Region("al_glom", 96, 0, 1.0),
    Region("al_local", 32, 32, 4.8),
    Region("al_proj", 64, 0, 5.0),
    Region("kc", 2048, 0, -1.0),
    Region("apl", 32, 32, 4.0),
    Region("mbon", 64, 16, 4.0),
    Region("dan", 32, 0, 4.0),
    Region("lh", 96, 38, 4.7),
    Region("cx", 128, 54, 4.7),
    Region("dn", 96, 24, 4.7),
    *(Region(f"vnc_{leg}", 128, 42, 5.0) for leg in ("lf", "lm", "lh", "rf", "rm", "rh")),
    Region("seg", 48, 19, 4.2),
    Region("value", 48, 19, 4.5),
    Region("comparator", 64, 26, 4.3),
)

MOTOR_LEARNING_REGIONS = (
    Region("proprio", 96, 0, 3.0),
    Region("flow", 48, 24, 3.0),
    Region("oct", 32, 0, 3.0),
)

MOTOR_LEARNING_PATHWAYS = (
    Pathway("flow", "value", 30, 0.9),
    Pathway("flow", "comparator", 30, 0.5),
    Pathway("flow", "cx", 24, 2.5),
    Pathway("flow", "dn", 24, 2.5),
    Pathway("comparator", "oct", 45, 0.6),
    Pathway("oct", "oct", 20, 0.3),
)


PATHWAYS = (
    Pathway("al_glom", "al_glom", 80, 0.55),
    Pathway("al_local", "al_glom", 30, 0.9),
    Pathway("al_proj", "al_glom", 50, 0.35),
    Pathway("al_glom", "al_local", 90, 0.6),
    Pathway("al_local", "al_local", 25, 0.7),
    Pathway("al_proj", "al_local", 40, 0.35),
    Pathway("al_glom", "al_proj", 90, 3.5),
    Pathway("al_local", "al_proj", 28, 1.0),
    Pathway("al_proj", "al_proj", 55, 0.4),
    Pathway("al_glom", "kc", 7, 10.0),
    Pathway("apl", "kc", 4, 1.2),
    Pathway("kc", "apl", 450, 0.8),
    Pathway("apl", "apl", 20, 0.45),
    Pathway("kc", "mbon", 450, 0.8),
    Pathway("mbon", "mbon", 40, 0.4),
    Pathway("apl", "mbon", 10, 0.6),
    Pathway("mbon", "dan", 55, 0.55),
    Pathway("value", "dan", 40, 0.55),
    Pathway("comparator", "dan", 40, 0.45),
    Pathway("dan", "dan", 20, 0.4),
    Pathway("al_proj", "lh", 60, 0.55),
    Pathway("lh", "lh", 80, 0.55),
    Pathway("value", "lh", 40, 0.4),
    Pathway("al_proj", "cx", 50, 0.35),
    Pathway("mbon", "cx", 50, 0.45),
    Pathway("lh", "cx", 75, 0.5),
    Pathway("cx", "cx", 120, 0.65),
    Pathway("dn", "cx", 30, 0.35),
    Pathway("cx", "dn", 120, 0.6),
    Pathway("dn", "dn", 80, 0.45),
    Pathway("lh", "dn", 45, 0.35),
    Pathway("dn", "seg", 85, 0.5),
    Pathway("seg", "seg", 35, 0.5),
    Pathway("value", "seg", 30, 0.35),
    Pathway("seg", "value", 45, 0.5),
    Pathway("value", "value", 35, 0.5),
    Pathway("lh", "value", 70, 0.4),
    Pathway("dn", "comparator", 50, 0.4),
    Pathway("cx", "comparator", 60, 0.4),
)


def pathways():
    result = list(PATHWAYS)
    legs = ("lf", "lm", "lh", "rf", "rm", "rh")
    for leg in legs:
        result.append(Pathway("dn", f"vnc_{leg}", 80, 0.45))
        result.append(Pathway(f"vnc_{leg}", f"vnc_{leg}", 100, 0.65))
        result.append(Pathway(f"vnc_{leg}", "comparator", 25, 0.35))
    for a, b in (("lf", "lm"), ("lm", "lh"), ("rf", "rm"), ("rm", "rh"),
                 ("lf", "rf"), ("lm", "rm"), ("lh", "rh")):
        result.append(Pathway(f"vnc_{a}", f"vnc_{b}", 50, 0.35))
        result.append(Pathway(f"vnc_{b}", f"vnc_{a}", 50, 0.35))
    return tuple(result)
