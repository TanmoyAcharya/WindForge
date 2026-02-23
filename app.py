from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.aero import AeroParams
from windforge.controllers import MPPTParams
from windforge.sim_mppt import MPPTSimConfig, run_rotor_mppt_sim_profile
from windforge.wind import ConstantWind, StepGust, RampWind, SineWind

from windforge.metrics import compute_metrics

st.set_page_config(page_title="WindForge Simulator", layout="wide")
st.title("🌬️ WindForge — Wind Turbine MPPT Simulator")
st.caption("Cp(λ) aerodynamics + generator electromechanics + MPPT load control + gust profiles")

with st.sidebar:
    st.header("Wind Profile")
    profile = st.selectbox("Profile", ["Constant", "Step Gust", "Ramp", "Sine"])

    if profile == "Constant":
        v = st.slider("Wind speed (m/s)", 2.0, 20.0, 8.0, 0.5)
        wind = ConstantWind(v=v)
    elif profile == "Step Gust":
        v0 = st.slider("v0 (m/s)", 2.0, 20.0, 7.0, 0.5)
        v1 = st.slider("v1 (m/s)", 2.0, 25.0, 11.0, 0.5)
        t_step = st.slider("t_step (s)", 0.5, 30.0, 8.0, 0.5)
        wind = StepGust(v0=v0, v1=v1, t_step=t_step)
    elif profile == "Ramp":
        v0 = st.slider("v0 (m/s)", 2.0, 20.0, 6.0, 0.5)
        v1 = st.slider("v1 (m/s)", 2.0, 25.0, 12.0, 0.5)
        t0 = st.slider("t0 (s)", 0.0, 20.0, 2.0, 0.5)
        t1 = st.slider("t1 (s)", 1.0, 40.0, 12.0, 0.5)
        wind = RampWind(v0=v0, v1=v1, t0=t0, t1=t1)
    else:
        v_mean = st.slider("Mean wind (m/s)", 2.0, 20.0, 9.0, 0.5)
        amp = st.slider("Amplitude (m/s)", 0.0, 8.0, 2.0, 0.5)
        freq = st.slider("Frequency (Hz)", 0.01, 0.5, 0.08, 0.01)
        wind = SineWind(v_mean=v_mean, amp=amp, freq_hz=freq)

    st.divider()
    st.header("Simulation")
    t_end = st.slider("t_end (s)", 5.0, 60.0, 20.0, 1.0)
    dt = st.select_slider("dt (s)", options=[0.01, 0.02, 0.05, 0.1], value=0.05)

    st.divider()
    st.header("Turbine + Generator")
    R = st.slider("Rotor radius R (m)", 0.5, 3.0, 1.5, 0.1)
    I = st.slider("Inertia I (kg·m²)", 0.01, 0.5, 0.12, 0.01)
    b = st.slider("Viscous friction b", 0.0, 0.05, 0.01, 0.001)
    tau_c = st.slider("Coulomb friction τc", 0.0, 0.2, 0.05, 0.01)

    R_g = st.slider("Generator R_g (Ω)", 0.05, 2.0, 0.5, 0.05)
    L = st.slider("Generator L (H)", 0.001, 0.2, 0.02, 0.001)
    k_e = st.slider("Back-EMF k_e", 0.05, 0.5, 0.20, 0.01)
    k_t = st.slider("Torque const k_t", 0.05, 0.5, 0.20, 0.01)

    st.divider()
    st.header("MPPT Controller")
    lambda_opt = st.slider("λ_opt", 3.0, 14.0, 8.0, 0.5)
    Kp = st.slider("Kp", 0.0, 0.3, 0.06, 0.005)
    Ki = st.slider("Ki", 0.0, 0.5, 0.12, 0.01)
    R_min = st.slider("R_min (Ω)", 0.1, 10.0, 1.0, 0.1)
    R_max = st.slider("R_max (Ω)", 5.0, 60.0, 25.0, 1.0)
    R_bias = st.slider("R_bias (Ω)", 0.1, 20.0, 6.0, 0.1)

run = st.button("▶ Run simulation", type="primary")

if run:
    p = RotorParams(I=I, b=b, tau_c=tau_c, k_wind=0.6)
    g = GeneratorParams(R_g=R_g, L=L, k_e=k_e, k_t=k_t, R_load=R_bias)
    a = AeroParams(rho=1.225, R=R, beta=0.0)
    mp = MPPTParams(lambda_opt=lambda_opt, Kp=Kp, Ki=Ki, R_min=R_min, R_max=R_max, R_bias=R_bias)
    cfg = MPPTSimConfig(t_end=t_end, dt=dt)

    res = run_rotor_mppt_sim_profile(wind=wind, p=p, g=g, a=a, mp=mp, cfg=cfg)
    v_arr = np.array([wind(t) for t in res.t], dtype=float)

    m = compute_metrics(
        t=res.t,
        lambda_ts=res.lambda_ts,
        cp=res.cp,
        power_load=res.power_load,
        lambda_opt=mp.lambda_opt,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Energy (Wh)", f"{m.energy_Wh:.2f}")
    c2.metric("Avg Cp", f"{m.avg_cp:.3f}")
    c3.metric("RMS λ error", f"{m.rms_lambda_error:.3f}")
    c4.metric("Settling time (s)", "N/A" if m.settling_time_s is None else f"{m.settling_time_s:.2f}")

    # Plots
    fig1 = plt.figure()
    plt.plot(res.t, v_arr)
    plt.title("Wind speed (m/s)")
    plt.grid(True)
    st.pyplot(fig1, clear_figure=True)

    fig2 = plt.figure()
    plt.plot(res.t, res.lambda_ts, label="lambda")
    plt.axhline(lambda_opt, linestyle="--", label="lambda_opt")
    plt.title("Tip-speed ratio λ")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig2, clear_figure=True)

    fig3 = plt.figure()
    plt.plot(res.t, res.cp)
    plt.title("Power coefficient Cp")
    plt.grid(True)
    st.pyplot(fig3, clear_figure=True)

    fig4 = plt.figure()
    plt.plot(res.t, res.R_load)
    plt.title("R_load command (Ω)")
    plt.grid(True)
    st.pyplot(fig4, clear_figure=True)

    fig5 = plt.figure()
    plt.plot(res.t, res.power_load)
    plt.title("Load power (W)")
    plt.grid(True)
    st.pyplot(fig5, clear_figure=True)

    # Download CSV
    df = pd.DataFrame({
        "t": res.t,
        "v_wind": v_arr,
        "theta": res.theta,
        "omega": res.omega,
        "i": res.i,
        "z": res.z,
        "R_load": res.R_load,
        "omega_ref": res.omega_ref,
        "lambda": res.lambda_ts,
        "cp": res.cp,
        "power_load": res.power_load,
    })
    st.download_button(
        "⬇ Download results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="windforge_results.csv",
        mime="text/csv",
    )
else:
    st.info("Set parameters in the sidebar, then click **Run simulation**.")