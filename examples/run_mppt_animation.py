from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.aero import AeroParams
from windforge.controllers import MPPTParams
from windforge.sim_mppt import MPPTSimConfig, run_rotor_mppt_sim


def main() -> None:
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1, k_wind=0.6)
    g = GeneratorParams(R_g=0.6, L=0.02, k_e=0.25, k_t=0.25, R_load=6.0)
    a = AeroParams(rho=1.225, R=0.8, beta=0.0)
    mp = MPPTParams(lambda_opt=8.0, Kp=0.06, Ki=0.12, R_min=1.0, R_max=25.0, R_bias=6.0)
    cfg = MPPTSimConfig(t_end=20.0, dt=0.05)

    v_wind = 8.0
    res = run_rotor_mppt_sim(v_wind=v_wind, p=p, g=g, a=a, mp=mp, cfg=cfg)

    # Fake rotor "blade" length
    L = a.R

    fig = plt.figure(figsize=(10, 6))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_lam = fig.add_subplot(2, 2, 2)
    ax_cp = fig.add_subplot(2, 2, 3)
    ax_R = fig.add_subplot(2, 2, 4)

    # 3D axis setup
    ax3d.set_title("Rotor animation")
    ax3d.set_xlim(-L, L)
    ax3d.set_ylim(-L, L)
    ax3d.set_zlim(-L, L)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # Line objects
    rotor_line, = ax3d.plot([0, L], [0, 0], [0, 0], lw=3)
    lam_line, = ax_lam.plot([], [], lw=2, label="lambda(t)")
    cp_line, = ax_cp.plot([], [], lw=2, label="Cp(t)")
    r_line, = ax_R.plot([], [], lw=2, label="R_load(t)")

    # Plot setups
    ax_lam.set_title("Tip-speed ratio λ")
    ax_lam.set_xlim(res.t[0], res.t[-1])
    ax_lam.set_ylim(0, max(12, float(np.max(res.lambda_ts) * 1.2)))
    ax_lam.grid(True)
    ax_lam.legend()

    ax_cp.set_title("Power coefficient Cp")
    ax_cp.set_xlim(res.t[0], res.t[-1])
    ax_cp.set_ylim(0, 0.65)
    ax_cp.grid(True)
    ax_cp.legend()

    ax_R.set_title("MPPT load command R_load")
    ax_R.set_xlim(res.t[0], res.t[-1])
    ax_R.set_ylim(0, max(30, float(np.max(res.R_load) * 1.2)))
    ax_R.grid(True)
    ax_R.legend()

    def update(k: int):
        # Use theta = integral omega dt (we didn't store explicitly in MPPT result, but we did)
        theta = float(res.theta[k])

        # Rotor line rotates in XY plane
        x2 = L * np.cos(theta)
        y2 = L * np.sin(theta)
        rotor_line.set_data([0, x2], [0, y2])
        rotor_line.set_3d_properties([0, 0])

        # Update live plots up to k
        t = res.t[: k + 1]
        lam_line.set_data(t, res.lambda_ts[: k + 1])
        cp_line.set_data(t, res.cp[: k + 1])
        r_line.set_data(t, res.R_load[: k + 1])

        return rotor_line, lam_line, cp_line, r_line

    ani = FuncAnimation(fig, update, frames=len(res.t), interval=30, blit=False)
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    input("Press Enter to close plots...")


if __name__ == "__main__":
    main()