from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .rotor import RotorParams, rotor_ode, tau_wind_simple, tau_load_simple


@dataclass(frozen=True)
class SimConfig:
    t_end: float = 20.0
    dt: float = 0.01
    theta0: float = 0.0
    omega0: float = 0.0


@dataclass(frozen=True)
class SimResult:
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    tau_wind: np.ndarray
    tau_load: np.ndarray
    power_mech: np.ndarray


def run_rotor_sim(v_wind: float, p: RotorParams, cfg: SimConfig) -> SimResult:
    n = int(round(cfg.t_end / cfg.dt)) + 1
    t_eval = np.linspace(0.0, cfg.t_end, n, dtype=float)
    y0 = np.array([cfg.theta0, cfg.omega0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rotor_ode(t, y, v_wind=v_wind, p=p),
        t_span=(0.0, cfg.t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    theta = sol.y[0]
    omega = sol.y[1]

    tau_w = np.array([tau_wind_simple(w, v_wind, p) for w in omega], dtype=float)
    tau_l = np.array([tau_load_simple(w, p) for w in omega], dtype=float)
    power = tau_l * omega

    return SimResult(sol.t, theta, omega, tau_w, tau_l, power)