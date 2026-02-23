from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.sim import SimConfig, run_rotor_gen_sim


def steady_mean(x: np.ndarray) -> float:
    n = len(x)
    i0 = int(0.8 * n)  # last 20% window
    return float(np.mean(x[i0:]))


def eval_power_for_Rload(v_wind: float, R_load: float) -> float:
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1, k_wind=0.6)

    g = GeneratorParams(R_g=0.6, L=0.02, k_e=0.25, k_t=0.25, R_load=float(R_load))
    cfg = SimConfig(t_end=25.0, dt=0.01, omega0=0.0, i0=0.0)

    res = run_rotor_gen_sim(v_wind=v_wind, p=p, g=g, cfg=cfg)

    # steady window
    n = len(res.t)
    i0 = int(0.8 * n)
    omega_ss = float(np.mean(res.omega[i0:]))
    i_ss = float(np.mean(res.i[i0:]))

    # objective = steady load power
    P = float(np.mean(res.power_load[i0:]))

    # engineering limits
    omega_max = 200.0  # rad/s
    i_max = 20.0       # A

    # penalty (quadratic) if exceed limits
    penalty = 0.0
    if abs(omega_ss) > omega_max:
        penalty += (abs(omega_ss) - omega_max) ** 2
    if abs(i_ss) > i_max:
        penalty += (abs(i_ss) - i_max) ** 2

    # subtract penalty scaled to watts
    return P - 50.0 * penalty


def main() -> None:
    v_wind = 8.0

    # 1) Coarse grid search (robust + easy)
    R_grid = np.linspace(1.0, 25.0, 25)  # ohms
    P_grid = np.array([eval_power_for_Rload(v_wind, R) for R in R_grid], dtype=float)

    best_idx = int(np.argmax(P_grid))
    best_R = float(R_grid[best_idx])
    best_P = float(P_grid[best_idx])

    print("=== WindForge Load Optimization ===")
    print(f"Wind speed: {v_wind:.2f} m/s")
    print(f"Best R_load (grid): {best_R:.3f} ohm")
    print(f"Best steady P_load: {best_P:.3f} W")

    # Plot power vs R_load
    plt.figure()
    plt.plot(R_grid, P_grid, marker="o")
    plt.xlabel("R_load (ohm)")
    plt.ylabel("Steady electrical power P_load (W)")
    plt.title("Optimization target: maximize steady load power")
    plt.grid(True)
    plt.show()

    # 2) Run a final simulation at best_R and show summary prints
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1, k_wind=0.6)
    g = GeneratorParams(R_g=0.6, L=0.02, k_e=0.25, k_t=0.25, R_load=best_R)
    cfg = SimConfig(t_end=25.0, dt=0.01, omega0=0.0, i0=0.0)

    res = run_rotor_gen_sim(v_wind=v_wind, p=p, g=g, cfg=cfg)

    omega_ss = steady_mean(res.omega)
    i_ss = steady_mean(res.i)
    print("--- Steady-state (last 20%) ---")
    print(f"omega_ss = {omega_ss:.3f} rad/s")
    print(f"i_ss     = {i_ss:.3f} A")
    print(f"P_load   = {steady_mean(res.power_load):.3f} W")


if __name__ == "__main__":
    main()