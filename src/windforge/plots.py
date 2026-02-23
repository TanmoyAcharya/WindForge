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