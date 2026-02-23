from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.sim import SimConfig, run_rotor_gen_sim


def steady_stats(res, frac: float = 0.8) -> dict:
    n = len(res.t)
    i0 = int(frac * n)
    omega_ss = float(np.mean(res.omega[i0:]))
    i_ss = float(np.mean(res.i[i0:]))
    p_load_ss = float(np.mean(res.power_load[i0:]))
    p_em_ss = float(np.mean(res.power_em[i0:]))
    return {
        "omega_ss": omega_ss,
        "i_ss": i_ss,
        "p_load_ss": p_load_ss,
        "p_em_ss": p_em_ss,
    }


def score_with_penalty(stats: dict, omega_max: float, i_max: float) -> float:
    P = stats["p_load_ss"]
    omega_ss = stats["omega_ss"]
    i_ss = stats["i_ss"]

    penalty = 0.0
    if abs(omega_ss) > omega_max:
        penalty += (abs(omega_ss) - omega_max) ** 2
    if abs(i_ss) > i_max:
        penalty += (abs(i_ss) - i_max) ** 2

    return P - 50.0 * penalty


def main() -> None:
    # Fixed parameters
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1, k_wind=0.6)
    base_g = GeneratorParams(R_g=0.6, L=0.02, k_e=0.25, k_t=0.25, R_load=5.0)
    cfg = SimConfig(t_end=25.0, dt=0.01, omega0=0.0, i0=0.0)

    # Engineering constraints
    omega_max = 200.0  # rad/s
    i_max = 20.0       # A

    # Wind speeds to evaluate
    v_list = np.linspace(3.0, 15.0, 13)

    # Candidate loads (grid). You can densify later.
    R_grid = np.linspace(1.0, 25.0, 25)

    rows = []
    for v in v_list:
        best = None

        for R in R_grid:
            g = GeneratorParams(
                R_g=base_g.R_g,
                L=base_g.L,
                k_e=base_g.k_e,
                k_t=base_g.k_t,
                R_load=float(R),
            )
            res = run_rotor_gen_sim(v_wind=float(v), p=p, g=g, cfg=cfg)
            stats = steady_stats(res, frac=0.8)

            score = score_with_penalty(stats, omega_max=omega_max, i_max=i_max)
            if (best is None) or (score > best["score"]):
                best = {
                    "v_wind_mps": float(v),
                    "R_load_ohm": float(R),
                    "score": float(score),
                    **stats,
                }

        rows.append(best)
        print(f"v={v:.1f} m/s -> R_opt={best['R_load_ohm']:.2f} ohm, P_load={best['p_load_ss']:.1f} W")

    df = pd.DataFrame(rows)

    out_csv = "opt_map_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}\n")
    print(df)

    # Plots
    plt.figure()
    plt.plot(df["v_wind_mps"], df["R_load_ohm"], marker="o")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Optimal R_load (ohm)")
    plt.title("Optimal load resistance vs wind speed")
    plt.grid(True)

    plt.figure()
    plt.plot(df["v_wind_mps"], df["p_load_ss"], marker="o")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Steady load power (W)")
    plt.title("Optimal steady electrical power vs wind speed")
    plt.grid(True)

    plt.figure()
    plt.plot(df["v_wind_mps"], np.abs(df["i_ss"]), marker="o")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("|Steady current| (A)")
    plt.title("Steady current at optimum vs wind speed")
    plt.grid(True)

    plt.figure()
    plt.plot(df["v_wind_mps"], df["omega_ss"], marker="o")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Steady ω (rad/s)")
    plt.title("Steady rotor speed at optimum vs wind speed")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()