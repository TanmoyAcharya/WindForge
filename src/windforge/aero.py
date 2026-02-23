from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AeroParams:
    rho: float = 1.225      # kg/m^3
    R: float = 1.8          # rotor radius (m)
    beta: float = 0.0       # pitch angle (deg)
    omega_eps: float = 1e-3 # avoid division by zero


def cp_generic(lambda_ts: float, beta_deg: float) -> float:
    """
    Common generic Cp(lambda, beta) approximation used in turbine literature.
    Output Cp clipped to [0, 0.59].
    """
    lam = max(float(lambda_ts), 1e-6)
    beta = float(beta_deg)

    denom = (1.0 / (lam + 0.08 * beta)) - (0.035 / (beta ** 3 + 1.0))
    lambda_i = 1.0 / max(denom, 1e-6)

    cp = 0.22 * (116.0 / lambda_i - 0.4 * beta - 5.0) * np.exp(-12.5 / lambda_i)
    return float(np.clip(cp, 0.0, 0.59))


def aero_torque_cp(omega: float, v_wind: float, a: AeroParams) -> tuple[float, float, float]:
    """
    Smooth startup + Cp(lambda) in operating region.
    Returns: (tau_wind, Cp, lambda_ts)
    """
    v = max(float(v_wind), 1e-6)
    w = float(omega)

    lam = abs(w) * a.R / v
    cp = cp_generic(lam, a.beta)

    A = np.pi * a.R**2
    P = 0.5 * a.rho * A * cp * v**3

    # Operating-region torque
    tau_cp = P / max(abs(w), a.omega_eps)

    # Gentle startup assist (scaled down, smooth fade-out)
    # This gives some push at very low speed without exploding dynamics.
    tau_start = 0.06 * a.rho * np.pi * a.R**3 * v**2

    w0 = 2.0  # rad/s transition speed
    blend = np.exp(-(abs(w) / w0) ** 2)   # ~1 near 0, ->0 as speed increases
    tau_mag = (blend * tau_start) + ((1.0 - blend) * tau_cp)

    sign = 1.0 if w >= 0.0 else -1.0
    return float(sign * tau_mag), float(cp), float(lam)