from __future__ import annotations
from windforge.rotor import RotorParams
from windforge.sim import SimConfig, run_rotor_sim
from windforge.plots import plot_timeseries


def main() -> None:
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1, k_wind=0.6, k_load=0.08)
    cfg = SimConfig(t_end=20.0, dt=0.01, omega0=0.0)

    v_wind = 8.0
    res = run_rotor_sim(v_wind=v_wind, p=p, cfg=cfg)

    print("Final omega (rad/s):", float(res.omega[-1]))
    print("Final power (W):", float(res.power_mech[-1]))

    plot_timeseries(res)


if __name__ == "__main__":
    main()