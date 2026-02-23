from __future__ import annotations

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.sim import SimConfig, run_rotor_gen_sim
from windforge.plots import plot_gen_timeseries


def main() -> None:
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1, k_wind=0.6)

    g = GeneratorParams(
        R_g=0.6,
        L=0.02,
        k_e=0.25,
        k_t=0.25,
        R_load=6.0,
    )

    cfg = SimConfig(t_end=25.0, dt=0.01, omega0=0.0, i0=0.0)

    v_wind = 8.0
    res = run_rotor_gen_sim(v_wind=v_wind, p=p, g=g, cfg=cfg)

    print("Final omega (rad/s):", float(res.omega[-1]))
    print("Final current (A):", float(res.i[-1]))
    print("Final P_load (W):", float(res.power_load[-1]))

    plot_gen_timeseries(res)


if __name__ == "__main__":
    main()