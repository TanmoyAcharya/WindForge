from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .generator import GeneratorParams
from .aero import AeroParams, aero_torque_cp
from .controllers import MPPTParams, mppt_rload
from .drivetrain import TwoMassParams

@dataclass(frozen=True)
class RotorParams:
    I: float = 0.25
    b: float = 0.01

    tau_c: float = 0.05
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
    tau_em = -g.k_t * i  # electromagnetic torque opposing motion

    # Rotor dynamics
    domega = (tw - tau_em - tau_visc - tau_coul) / p.I
    dtheta = omega

    # Electrical dynamics: L di/dt = -(R_g + R_load)i - e
    R_total = g.R_g + g.R_load
    di = (-(R_total * i) - e) / g.L

    return np.array([dtheta, domega, di], dtype=float)

def rotor_gen_cp_ode(t, y, v_wind, p, g, a: AeroParams):
    """
    y = [theta, omega, i]
    uses Cp(lambda) aero torque.
    """
    theta = float(y[0])
    omega = float(y[1])
    i = float(y[2])

    tw, cp, lam = aero_torque_cp(omega=omega, v_wind=v_wind, a=a)

    tau_visc = p.b * omega
    tau_coul = p.tau_c * smooth_sign(omega, p.omega_eps)

    e = g.k_e * omega
    tau_em = -g.k_t * i

    domega = (tw - tau_em - tau_visc - tau_coul) / p.I
    dtheta = omega

    R_total = g.R_g + g.R_load
    di = (-(R_total * i) - e) / g.L

    return np.array([dtheta, domega, di], dtype=float)

def rotor_gen_cp_mppt_ode(t, y, v_wind, p, g, a, mp: MPPTParams):
    """
    y = [theta, omega, i, z]
      z = integral of speed error (omega_ref - omega)
    """
    theta = float(y[0])
    omega = float(y[1])
    i = float(y[2])
    z = float(y[3])

    # Aero torque from Cp(lambda)
    tw, cp, lam = aero_torque_cp(omega=omega, v_wind=v_wind, a=a)

    # MPPT PI controller chooses R_load
    R_load, omega_ref, e = mppt_rload(omega=omega, v_wind=v_wind, rotor_R=a.R, z_int=z, mp=mp)

    # Mechanical frictions
    tau_visc = p.b * omega
    tau_coul = p.tau_c * smooth_sign(omega, p.omega_eps)

    # Generator electromechanics
    e_back = g.k_e * omega
    tau_em = -g.k_t * i

    domega = (tw - tau_em - tau_visc - tau_coul) / p.I
    dtheta = omega

    # Electrical dynamics with controlled load
    R_total = g.R_g + R_load
    di = (-(R_total * i) - e_back) / g.L

    # Integral of speed error
    dz = e

    return np.array([dtheta, domega, di, dz], dtype=float)
def rotor_2mass_cp_mppt_ode(t, y, wind_fn, p: RotorParams, d: TwoMassParams, g, a: AeroParams, mp: MPPTParams):
    """
    State y = [theta_r, omega_r, theta_g, omega_g, i, z]
    wind_fn(t) -> v_wind (m/s)
    """
    theta_r = float(y[0])
    omega_r = float(y[1])
    theta_g = float(y[2])
    omega_g = float(y[3])
    i = float(y[4])
    z = float(y[5])

    v_wind = float(wind_fn(t))

    # Aero torque uses rotor-side speed
    tau_aero, cp, lam = aero_torque_cp(omega=omega_r, v_wind=v_wind, a=a)

    # MPPT targets generator-side speed (what you can "sense")
    R_load, omega_ref, e = mppt_rload(omega=omega_g, v_wind=v_wind, rotor_R=a.R, z_int=z, mp=mp)

    # Shaft torque (positive transfers from rotor -> generator)
    twist = theta_r - theta_g
    relw = omega_r - omega_g
    tau_shaft = d.k_s * twist + d.c_s * relw

    # Rotor losses
    tau_visc = p.b * omega_r
    tau_coul = p.tau_c * smooth_sign(omega_r, p.omega_eps)

    # Electromechanical torque on generator
    tau_em = -g.k_t * i

    # Rotor dynamics
    domega_r = (tau_aero - tau_shaft - tau_visc - tau_coul) / d.J_r
    dtheta_r = omega_r

    # Generator dynamics
    domega_g = (tau_shaft - tau_em) / d.J_g
    dtheta_g = omega_g

    # Electrical dynamics (back-EMF uses generator speed)
    e_back = g.k_e * omega_g
    R_total = g.R_g + R_load
    di = (-(R_total * i) - e_back) / g.L

    # MPPT integrator
    dz = e

    import numpy as np
    return np.array([dtheta_r, domega_r, dtheta_g, domega_g, di, dz], dtype=float)