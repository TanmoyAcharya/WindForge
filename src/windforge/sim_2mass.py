from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .rotor import RotorParams, rotor_2mass_cp_mppt_ode
from .generator import GeneratorParams
from .aero import AeroParams, aero_torque_cp
from .controllers import MPPTParams, mppt_rload
from .drivetrain import TwoMassParams
from .wind import WindProfile


@dataclass(frozen=True)
class TwoMassSimConfig:
    t_end: float = 20.0
    dt: float = 0.05
    theta_r0: float = 0.0
    omega_r0: float = 0.0
    theta_g0: float = 0.0
    omega_g0: float = 0.0
    i0: float = 0.0
    z0: float = 0.0


@dataclass(frozen=True)
class TwoMassSimResult:
    t: np.ndarray
    theta_r: np.ndarray
    omega_r: np.ndarray
    theta_g: np.ndarray
    omega_g: np.ndarray
    i: np.ndarray
    z: np.ndarray
    v_wind: np.ndarray
    R_load: np.ndarray
    omega_ref: np.ndarray
    cp: np.ndarray
    lambda_ts: np.ndarray
    tau_shaft: np.ndarray
    power_load: np.ndarray


def run_2mass_mppt_sim(
    wind: WindProfile,
    p: RotorParams,
    d: TwoMassParams,
    g: GeneratorParams,
    a: AeroParams,
    mp: MPPTParams,
    cfg: TwoMassSimConfig,
) -> TwoMassSimResult:
    n = int(round(cfg.t_end / cfg.dt)) + 1
    t_eval = np.linspace(0.0, cfg.t_end, n, dtype=float)

    y0 = np.array([cfg.theta_r0, cfg.omega_r0, cfg.theta_g0, cfg.omega_g0, cfg.i0, cfg.z0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rotor_2mass_cp_mppt_ode(t, y, wind_fn=wind, p=p, d=d, g=g, a=a, mp=mp),
        t_span=(0.0, cfg.t_end),
        y0=y0,
        t_eval=t_eval,
        rtol=1e-5,
        atol=1e-7,
        max_step=cfg.dt,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    theta_r, omega_r, theta_g, omega_g, cur, z = sol.y
    v_arr = np.array([float(wind(t)) for t in sol.t], dtype=float)

    R_arr = np.zeros_like(omega_g)
    wref_arr = np.zeros_like(omega_g)
    cp_arr = np.zeros_like(omega_r)
    lam_arr = np.zeros_like(omega_r)
    tau_shaft = np.zeros_like(omega_r)

    for k in range(len(sol.t)):
        # controller (based on generator speed)
        Rk, wref, _ = mppt_rload(float(omega_g[k]), float(v_arr[k]), a.R, float(z[k]), mp)
        R_arr[k] = Rk
        wref_arr[k] = wref

        # aero quantities (based on rotor speed)
        _, cp, lam = aero_torque_cp(float(omega_r[k]), float(v_arr[k]), a)
        cp_arr[k] = cp
        lam_arr[k] = lam

        # shaft torque
        twist = float(theta_r[k] - theta_g[k])
        relw = float(omega_r[k] - omega_g[k])
        tau_shaft[k] = d.k_s * twist + d.c_s * relw

    power_load = (cur ** 2) * R_arr

    return TwoMassSimResult(
        t=sol.t,
        theta_r=theta_r,
        omega_r=omega_r,
        theta_g=theta_g,
        omega_g=omega_g,
        i=cur,
        z=z,
        v_wind=v_arr,
        R_load=R_arr,
        omega_ref=wref_arr,
        cp=cp_arr,
        lambda_ts=lam_arr,
        tau_shaft=tau_shaft,
        power_load=power_load,
    )