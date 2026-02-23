from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MPPTParams:
    lambda_opt: float = 8.0
    Kp: float = 0.06
    Ki: float = 0.12
    R_min: float = 1.0
    R_max: float = 25.0
    R_bias: float = 6.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def mppt_rload(omega: float, v_wind: float, rotor_R: float, z_int: float, mp: MPPTParams) -> tuple[float, float, float]:
    """
    Returns: (R_load, omega_ref, error)
    """
    omega_ref = mp.lambda_opt * max(v_wind, 1e-6) / max(rotor_R, 1e-6)
    e = omega_ref - omega
    R_cmd = mp.R_bias + mp.Kp * e + mp.Ki * z_int
    return clamp(R_cmd, mp.R_min, mp.R_max), float(omega_ref), float(e)