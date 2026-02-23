from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorParams:
    # Electrical parameters
    R_g: float = 0.5      # internal resistance (ohm)
    L: float = 0.02       # inductance (H)

    # Electromechanical constants
    k_e: float = 0.2      # back-EMF constant (V per rad/s)
    k_t: float = 0.2      # torque constant (N*m per A)

    # Load
    R_load: float = 5.0   # resistive load (ohm)