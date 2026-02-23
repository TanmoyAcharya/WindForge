from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TwoMassParams:
    J_r: float = 0.12   # rotor inertia (kg m^2)
    J_g: float = 0.02   # generator inertia (kg m^2)
    k_s: float = 35.0   # shaft stiffness (N m/rad)
    c_s: float = 0.6    # shaft damping (N m s/rad)