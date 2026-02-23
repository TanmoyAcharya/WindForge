from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .generator import GeneratorParams

@dataclass(frozen=True)
class RotorParams:
    I: float = 0.25
    b: float = 0.02
    tau_c: float = 0.1
    k_wind: float = 0.6
    k_load: float = 0.08
    omega_eps: float = 1e-3


def smooth_sign(x: float, eps: float) -> float:
    return float(x / (abs(x) + eps))


def tau_wind_simple(omega: float, v_wind: float, p: RotorParams) -> float:
    alpha = 0.25
    return p.k_wind * (v_wind ** 2) / (1.0 + alpha * max(omega, 0.0))


def tau_load_simple(omega: float, p: RotorParams) -> float:
    return p.k_load * omega


def rotor_ode(t: float, y: np.ndarray, v_wind: float, p: RotorParams) -> np.ndarray:
    theta, omega = float(y[0]), float(y[1])

    tw = tau_wind_simple(omega=omega, v_wind=v_wind, p=p)
    tl = tau_load_simple(omega=omega, p=p)

    tau_visc = p.b * omega
    tau_coul = p.tau_c * smooth_sign(omega, p.omega_eps)

    domega = (tw - tl - tau_visc - tau_coul) / p.I
    dtheta = omega
    return np.array([dtheta, domega], dtype=float)


def rotor_gen_ode(t: float, y: np.ndarray, v_wind: float, p: RotorParams, g: GeneratorParams) -> np.ndarray:
    """
    State y = [theta, omega, i]
      theta: rotor angle (rad)
      omega: rotor speed (rad/s)
      i: generator current (A)
    """
    theta, omega, i = float(y[0]), float(y[1]), float(y[2])

    # Mechanical torques
    tw = tau_wind_simple(omega=omega, v_wind=v_wind, p=p)

    tau_visc = p.b * omega
    tau_coul = p.tau_c * smooth_sign(omega, p.omega_eps)

    # Generator electromechanics
    e = g.k_e * omega
    tau_em = g.k_t * i  # electromagnetic torque opposing motion

    # Rotor dynamics
    domega = (tw - tau_em - tau_visc - tau_coul) / p.I
    dtheta = omega

    # Electrical dynamics: L di/dt = -(R_g + R_load)i - e
    R_total = g.R_g + g.R_load
    di = (-(R_total * i) - e) / g.L

    return np.array([dtheta, domega, di], dtype=float)