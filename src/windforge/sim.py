from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .rotor import RotorParams, rotor_gen_ode, tau_wind_simple
from .generator import GeneratorParams


# =============================
# Coupled electro-mechanical sim
# =============================

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


# =============================
# Legacy mechanical-only sim API
# (kept for old examples)
# =============================

@dataclass(frozen=True)
class LegacyRotorParams:
    I: float = 0.25
    b: float = 0.02
    tau_c: float = 0.1
    k_wind: float = 0.6
    k_load: float = 0.08
    omega_eps: float = 1e-3


@dataclass(frozen=True)
class LegacySimResult:
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    tau_wind: np.ndarray
    tau_load: np.ndarray
    power_mech: np.ndarray


def run_rotor_sim(v_wind: float, p: LegacyRotorParams, cfg: SimConfig) -> LegacySimResult:
    """
    Mechanical-only simulation (legacy): kept so older examples still run.

    Uses:
      tau_wind_simple(...) for drive torque
      tau_load = k_load * omega * |omega| as simple load proxy
    """
    n = int(round(cfg.t_end / cfg.dt)) + 1
    t_eval = np.linspace(0.0, cfg.t_end, n, dtype=float)

    # Reuse RotorParams just to call tau_wind_simple safely
    p_wind = RotorParams(I=p.I, b=p.b, tau_c=p.tau_c, k_wind=p.k_wind)

    def ode(t: float, y: np.ndarray) -> np.ndarray:
        theta = float(y[0])
        omega = float(y[1])

        tw = tau_wind_simple(omega=omega, v_wind=v_wind, p=p_wind)
        tau_load = p.k_load * omega * abs(omega)

        tau_visc = p.b * omega
        tau_coul = p.tau_c * (omega / (abs(omega) + p.omega_eps))

        domega = (tw - tau_load - tau_visc - tau_coul) / p.I
        dtheta = omega
        return np.array([dtheta, domega], dtype=float)

    y0 = np.array([cfg.theta0, cfg.omega0], dtype=float)

    sol = solve_ivp(
        fun=ode,
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

    tw = np.array([tau_wind_simple(w, v_wind, p_wind) for w in omega], dtype=float)
    tau_load = p.k_load * omega * np.abs(omega)
    power_mech = (tw - tau_load) * omega

    return LegacySimResult(
        t=sol.t,
        theta=theta,
        omega=omega,
        tau_wind=tw,
        tau_load=tau_load,
        power_mech=power_mech,
    )