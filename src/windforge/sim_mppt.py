from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .rotor import RotorParams, rotor_gen_cp_mppt_ode
from .generator import GeneratorParams
from .aero import AeroParams, aero_torque_cp
from .controllers import MPPTParams, mppt_rload


@dataclass(frozen=True)
class MPPTSimConfig:
    t_end: float = 25.0
    dt: float = 0.01
    theta0: float = 0.0
    omega0: float = 0.0
    i0: float = 0.0
    z0: float = 0.0


@dataclass(frozen=True)
class MPPTSimResult:
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    i: np.ndarray
    z: np.ndarray
    R_load: np.ndarray
    omega_ref: np.ndarray
    tau_wind: np.ndarray
    cp: np.ndarray
    lambda_ts: np.ndarray
    power_load: np.ndarray


def run_rotor_mppt_sim(v_wind: float, p: RotorParams, g: GeneratorParams, a: AeroParams, mp: MPPTParams, cfg: MPPTSimConfig) -> MPPTSimResult:
    n = int(round(cfg.t_end / cfg.dt)) + 1
    t_eval = np.linspace(0.0, cfg.t_end, n, dtype=float)

    y0 = np.array([cfg.theta0, cfg.omega0, cfg.i0, cfg.z0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rotor_gen_cp_mppt_ode(t, y, v_wind=v_wind, p=p, g=g, a=a, mp=mp),
        t_span=(0.0, cfg.t_end),
        y0=y0,
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-7,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    theta, omega, cur, z = sol.y[0], sol.y[1], sol.y[2], sol.y[3]

    tau_w = np.zeros_like(omega)
    cp_arr = np.zeros_like(omega)
    lam_arr = np.zeros_like(omega)
    R_arr = np.zeros_like(omega)
    wref_arr = np.zeros_like(omega)

    for k in range(len(omega)):
        tw, cp, lam = aero_torque_cp(float(omega[k]), v_wind, a)
        Rk, wref, _ = mppt_rload(float(omega[k]), v_wind, a.R, float(z[k]), mp)
        tau_w[k] = tw
        cp_arr[k] = cp
        lam_arr[k] = lam
        R_arr[k] = Rk
        wref_arr[k] = wref

    power_load = (cur ** 2) * R_arr

    return MPPTSimResult(
        t=sol.t,
        theta=theta,
        omega=omega,
        i=cur,
        z=z,
        R_load=R_arr,
        omega_ref=wref_arr,
        tau_wind=tau_w,
        cp=cp_arr,
        lambda_ts=lam_arr,
        power_load=power_load,
    )