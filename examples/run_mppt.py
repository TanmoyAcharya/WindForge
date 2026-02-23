from __future__ import annotations
import numpy as np

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.aero import AeroParams
from windforge.controllers import MPPTParams
from windforge.sim_mppt import MPPTSimConfig, run_rotor_mppt_sim_profile
from windforge.wind import StepGust

from windforge.metrics import compute_metrics
from windforge.report import (
    save_timeseries_csv,
    save_metrics_json,
    save_summary_plots,
    write_report_md,
)


def main() -> None:
    wind = StepGust(v0=7.0, v1=11.0, t_step=8.0)

    p = RotorParams(I=0.12, b=0.01, tau_c=0.05, k_wind=0.6)
    g = GeneratorParams(R_g=0.5, L=0.02, k_e=0.20, k_t=0.20, R_load=6.0)
    a = AeroParams(rho=1.225, R=1.5, beta=0.0)
    mp = MPPTParams(lambda_opt=8.0, Kp=0.06, Ki=0.12, R_min=1.0, R_max=25.0, R_bias=6.0)

    cfg = MPPTSimConfig(t_end=20.0, dt=0.05)

    res = run_rotor_mppt_sim_profile(wind=wind, p=p, g=g, a=a, mp=mp, cfg=cfg)
    v_arr = np.array([wind(t) for t in res.t], dtype=float)

    metrics = compute_metrics(
        t=res.t,
        lambda_ts=res.lambda_ts,
        cp=res.cp,
        power_load=res.power_load,
        lambda_opt=mp.lambda_opt,
        tau_shaft=None,
    )

    name = "gust_mppt"
    csv_path = save_timeseries_csv("outputs", name, data={
        "t": res.t,
        "v_wind": v_arr,
        "theta": res.theta,
        "omega": res.omega,
        "i": res.i,
        "R_load": res.R_load,
        "omega_ref": res.omega_ref,
        "lambda": res.lambda_ts,
        "cp": res.cp,
        "power_load": res.power_load,
    })
    plot_path = save_summary_plots(
        "outputs",
        name,
        t=res.t,
        v_wind=v_arr,
        lambda_ts=res.lambda_ts,
        cp=res.cp,
        r_load=res.R_load,
        power_load=res.power_load,
        omega_r=res.omega,
        omega_g=None,
        omega_ref=res.omega_ref,
        tau_shaft=None,
        lambda_opt=mp.lambda_opt,
    )
    save_metrics_json("outputs", name, metrics)
    write_report_md(
        "outputs",
        name,
        title="WindForge — Gust MPPT Report",
        description="Step gust (7→11 m/s) with Cp(λ) aerodynamics + generator model + MPPT load control.",
        metrics=metrics,
        plot_path=plot_path,
        csv_path=csv_path,
    )

    print("Saved outputs to ./outputs/")
    print(" -", plot_path)
    print(" -", csv_path)


if __name__ == "__main__":
    main()