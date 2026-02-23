from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .rotor import RotorParams, rotor_gen_ode, tau_wind_simple
from .generator import GeneratorParams


@dataclass(frozen=True)
class SimConfig:
    t_end: float = 25.0
    dt: float = 0.01
    theta0: float = 0.0
    omega0: float = 0.0
    i0: float = 0.0


@dataclass(frozen=True)
class SimResult:
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    i: np.ndarray
    tau_wind: np.ndarray
    tau_em: np.ndarray
    power_em: np.ndarray
    power_load: np.ndarray
    power_copper: np.ndarray


def run_rotor_gen_sim(v_wind: float, p: RotorParams, g: GeneratorParams, cfg: SimConfig) -> SimResult:
    n = int(round(cfg.t_end / cfg.dt)) + 1
    t_eval = np.linspace(0.0, cfg.t_end, n, dtype=float)

    y0 = np.array([cfg.theta0, cfg.omega0, cfg.i0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rotor_gen_ode(t, y, v_wind=v_wind, p=p, g=g),
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
    cur = sol.y[2]

    tau_w = np.array([tau_wind_simple(w, v_wind, p) for w in omega], dtype=float)

    # Must match sign convention in rotor_gen_ode
    tau_em = -g.k_t * cur

    power_em = tau_em * omega
    power_load = (cur ** 2) * g.R_load
    power_copper = (cur ** 2) * g.R_g

    return SimResult(
        t=sol.t,
        theta=theta,
        omega=omega,
        i=cur,
        tau_wind=tau_w,
        tau_em=tau_em,
        power_em=power_em,
        power_load=power_load,
        power_copper=power_copper,
    )