from __future__ import annotations

from pathlib import Path
import sys

# --- MUST come before importing windforge ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Debug (keep for now, delete after it works) ---
import windforge  # noqa
st.sidebar.caption("🔧 Debug (remove later)")
st.sidebar.write("windforge from:", windforge.__file__)
st.sidebar.write("SRC exists:", SRC.exists())
st.sidebar.write("sys.path[0]:", sys.path[0])

# --- WindForge imports (ONLY ONCE) ---
from windforge.wind import ConstantWind, StepGust, RampWind, SineWind
from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.aero import AeroParams
from windforge.controllers import MPPTParams
from windforge.sim_mppt import MPPTSimConfig, run_rotor_mppt_sim_profile
from windforge.metrics import compute_metrics


st.set_page_config(page_title="WindForge Simulator", layout="wide")
st.title("🌬️ WindForge — Wind Turbine MPPT Simulator")
st.caption("Cp(λ) aerodynamics + generator electromechanics + MPPT load control + gust profiles")

# -------------------------
# Sidebar controls
# -------------------------
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

# -------------------------
# Main run
# -------------------------
# -------------------------
# Plot 1 — Wind speed
# -------------------------
fig_wind = go.Figure()
fig_wind.add_trace(go.Scatter(
    x=res.t,
    y=v_arr,
    mode="lines",
    name="Wind speed",
))
fig_wind.update_layout(
    title="Wind Speed (m/s)",
    xaxis_title="Time (s)",
    yaxis_title="Wind speed (m/s)",
    template="plotly_white",
)
st.plotly_chart(fig_wind, use_container_width=True)


# -------------------------
# Plot 2 — Tip-speed ratio
# -------------------------
fig_lambda = go.Figure()
fig_lambda.add_trace(go.Scatter(
    x=res.t,
    y=res.lambda_ts,
    mode="lines",
    name="λ"
))
fig_lambda.add_trace(go.Scatter(
    x=res.t,
    y=[lambda_opt]*len(res.t),
    mode="lines",
    name="λ_opt",
    line=dict(dash="dash")
))
fig_lambda.update_layout(
    title="Tip-Speed Ratio (λ)",
    xaxis_title="Time (s)",
    yaxis_title="λ",
    template="plotly_white",
)
st.plotly_chart(fig_lambda, use_container_width=True)


# -------------------------
# Plot 3 — Cp
# -------------------------
fig_cp = go.Figure()
fig_cp.add_trace(go.Scatter(
    x=res.t,
    y=res.cp,
    mode="lines",
    name="Cp"
))
fig_cp.update_layout(
    title="Power Coefficient Cp",
    xaxis_title="Time (s)",
    yaxis_title="Cp",
    template="plotly_white",
)
st.plotly_chart(fig_cp, use_container_width=True)


# -------------------------
# Plot 4 — R_load
# -------------------------
fig_r = go.Figure()
fig_r.add_trace(go.Scatter(
    x=res.t,
    y=res.R_load,
    mode="lines",
    name="R_load"
))
fig_r.update_layout(
    title="MPPT Load Resistance (Ω)",
    xaxis_title="Time (s)",
    yaxis_title="R_load (Ω)",
    template="plotly_white",
)
st.plotly_chart(fig_r, use_container_width=True)


# -------------------------
# Plot 5 — Power
# -------------------------
fig_power = go.Figure()
fig_power.add_trace(go.Scatter(
    x=res.t,
    y=res.power_load,
    mode="lines",
    name="Load Power"
))
fig_power.update_layout(
    title="Electrical Power Output (W)",
    xaxis_title="Time (s)",
    yaxis_title="Power (W)",
    template="plotly_white",
)
st.plotly_chart(fig_power, use_container_width=True)

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