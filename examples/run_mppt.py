from __future__ import annotations
import matplotlib.pyplot as plt

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.aero import AeroParams
from windforge.controllers import MPPTParams
from windforge.sim_mppt import MPPTSimConfig, run_rotor_mppt_sim


def main():
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1)
    g = GeneratorParams(R_g=0.6, L=0.02, k_e=0.25, k_t=0.25, R_load=6.0)
    a = AeroParams(rho=1.225, R=0.8, beta=0.0)
    mp = MPPTParams(lambda_opt=8.0, Kp=0.08, Ki=0.15, R_min=1.0, R_max=25.0, R_bias=6.0)
    cfg = MPPTSimConfig(t_end=25.0, dt=0.01)

    v_wind = 8.0
    res = run_rotor_mppt_sim(v_wind=v_wind, p=p, g=g, a=a, mp=mp, cfg=cfg)

    plt.figure()
    plt.plot(res.t, res.omega)
    plt.xlabel("t (s)")
    plt.ylabel("omega (rad/s)")
    plt.title("MPPT: rotor speed")
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.lambda_ts, label="lambda")
    plt.plot(res.t, [mp.lambda_opt]*len(res.t), "--", label="lambda_opt")
    plt.xlabel("t (s)")
    plt.ylabel("Tip-speed ratio")
    plt.title("MPPT: TSR tracking")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.cp)
    plt.xlabel("t (s)")
    plt.ylabel("Cp")
    plt.title("Cp over time")
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.R_load)
    plt.xlabel("t (s)")
    plt.ylabel("R_load (ohm)")
    plt.title("Control output: load resistance")
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.power_load)
    plt.xlabel("t (s)")
    plt.ylabel("P_load (W)")
    plt.title("Electrical power to load (MPPT)")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()