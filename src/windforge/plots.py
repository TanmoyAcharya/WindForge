from __future__ import annotations
import matplotlib.pyplot as plt
from .sim import SimResult


def plot_timeseries(res: SimResult) -> None:
    plt.figure()
    plt.plot(res.t, res.omega)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular speed ω (rad/s)")
    plt.title("Rotor speed vs time")
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.tau_wind, label="tau_wind")
    plt.plot(res.t, res.tau_load, label="tau_load")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N·m)")
    plt.title("Torques vs time")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.power_mech)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W) = τ_load · ω")
    plt.title("Mechanical power to load vs time")
    plt.grid(True)

    plt.show()

def plot_gen_timeseries(res: SimResult) -> None:

    plt.figure()
    plt.plot(res.t, res.omega)
    plt.xlabel("Time (s)")
    plt.ylabel("ω (rad/s)")
    plt.title("Rotor speed vs time")
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.i)
    plt.xlabel("Time (s)")
    plt.ylabel("Current i (A)")
    plt.title("Generator current vs time")
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.tau_wind, label="tau_wind")
    plt.plot(res.t, res.tau_em, label="tau_em")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N·m)")
    plt.title("Wind torque vs electromagnetic torque")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(res.t, res.power_em, label="P_em = tau_em * ω")
    plt.plot(res.t, res.power_load, label="P_load = i^2 * R_load")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Mechanical→Electrical power conversion")
    plt.legend()
    plt.grid(True)

    plt.show()