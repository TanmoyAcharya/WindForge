from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windforge.rotor import RotorParams
from windforge.sim import SimConfig, run_rotor_sim


def main() -> None:
    # --- model parameters ---
    p = RotorParams(
        I=0.25,
        b=0.02,
        tau_c=0.1,
        k_wind=0.6,
        k_load=0.08,
    )
    cfg = SimConfig(t_end=30.0, dt=0.01, omega0=0.0)

    # wind speed sweep (m/s)
    v_list = np.linspace(3.0, 15.0, 13)

    rows = []
    for v in v_list:
        res = run_rotor_sim(v_wind=float(v), p=p, cfg=cfg)

        # take the last 20% of samples as "steady-ish" window
        n = len(res.t)
        i0 = int(0.8 * n)

        omega_ss = float(np.mean(res.omega[i0:]))
        power_ss = float(np.mean(res.power_mech[i0:]))
        power_peak = float(np.max(res.power_mech))
        omega_peak = float(np.max(res.omega))

        rows.append(
            {
                "v_wind_mps": float(v),
                "omega_ss_radps": omega_ss,
                "power_ss_W": power_ss,
                "power_peak_W": power_peak,
                "omega_peak_radps": omega_peak,
            }
        )

    df = pd.DataFrame(rows)

    # Save results
    out_csv = "wind_sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df)

    # Plot power curve
    plt.figure()
    plt.plot(df["v_wind_mps"], df["power_ss_W"], marker="o")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Steady power to load (W)")
    plt.title("WindForge — Power curve (steady-state)")
    plt.grid(True)

    plt.figure()
    plt.plot(df["v_wind_mps"], df["omega_ss_radps"], marker="o")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Steady rotor speed (rad/s)")
    plt.title("WindForge — Steady rotor speed vs wind")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()