from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.aero import AeroParams
from windforge.controllers import MPPTParams
from windforge.drivetrain import TwoMassParams
from windforge.wind import StepGust
from windforge.sim_2mass import TwoMassSimConfig, run_2mass_mppt_sim


def main() -> None:
    wind = StepGust(v0=7.0, v1=11.0, t_step=8.0)

    p = RotorParams(I=0.25, b=0.01, tau_c=0.05, k_wind=0.6)
    d = TwoMassParams(J_r=0.12, J_g=0.02, k_s=35.0, c_s=0.6)
    g = GeneratorParams(R_g=0.5, L=0.02, k_e=0.20, k_t=0.20, R_load=6.0)
    a = AeroParams(rho=1.225, R=1.5, beta=0.0)
    mp = MPPTParams(lambda_opt=8.0, Kp=0.06, Ki=0.12, R_min=1.0, R_max=25.0, R_bias=6.0)

    cfg = TwoMassSimConfig(t_end=20.0, dt=0.05)

    res = run_2mass_mppt_sim(wind=wind, p=p, d=d, g=g, a=a, mp=mp, cfg=cfg)

    plt.figure()
    plt.plot(res.t, res.v_wind)
    plt.title("Wind speed")
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.omega_r, label="omega_rotor")
    plt.plot(res.t, res.omega_g, label="omega_gen")
    plt.plot(res.t, res.omega_ref, "--", label="omega_ref")
    plt.title("Two-mass speeds")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.tau_shaft)
    plt.title("Shaft torque (torsional load)")
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.lambda_ts, label="lambda")
    plt.plot(res.t, res.cp, label="Cp")
    plt.title("Aerodynamics")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()