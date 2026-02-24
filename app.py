from __future__ import annotations

from pathlib import Path
import sys
import io
import json
import zipfile

# --- MUST come before importing windforge ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Session memory for comparisons
if "runs" not in st.session_state:
    st.session_state.runs = {}

# --- Debug (remove later) ---
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

with st.expander("Model assumptions / notes"):
    st.markdown(
        """
- Aerodynamics: Cp(λ) parametric model (β fixed).
- Drivetrain: rigid rotor inertia + viscous + Coulomb friction.
- Generator: RL + back-EMF, torque proportional to current.
- Control: MPPT via λ tracking; load represented as variable resistance.
- Efficiency shown is a proxy (based on modeled powers).
        """
    )

tab_dash, tab_analysis, tab_compare, tab_sweep, tab_export = st.tabs(
    ["📌 Dashboard", "🔬 Analysis", "📊 Compare", "⚡ Sweep", "📄 Export"]
)

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

    st.divider()
    st.header("Run management")
    run_label = st.text_input("Run label", value=f"Run {len(st.session_state.runs)+1}")
    clear = st.button("🗑️ Clear saved runs")
    if clear:
        st.session_state.runs = {}
        st.success("Cleared saved runs.")

run = st.button("▶ Run simulation", type="primary")

# Build params (used by sim + sweep)
p = RotorParams(I=I, b=b, tau_c=tau_c, k_wind=0.6)
g = GeneratorParams(R_g=R_g, L=L, k_e=k_e, k_t=k_t, R_load=R_bias)
a = AeroParams(rho=1.225, R=R, beta=0.0)
mp = MPPTParams(lambda_opt=lambda_opt, Kp=Kp, Ki=Ki, R_min=R_min, R_max=R_max, R_bias=R_bias)
cfg = MPPTSimConfig(t_end=t_end, dt=dt)

# -------------------------
# RUN SIMULATION
# -------------------------
if run:
    with st.spinner("Running simulation..."):
        res = run_rotor_mppt_sim_profile(wind=wind, p=p, g=g, a=a, mp=mp, cfg=cfg)

    v_arr = np.array([float(wind(t)) for t in res.t], dtype=float)

    # --- Derived quantities (engineering outputs) ---
    R_load_ts = res.R_load
    i_ts = res.i
    omega_ts = res.omega

    v_load = i_ts * R_load_ts
    p_load = res.power_load
    p_copper = (i_ts ** 2) * g.R_g

    tau_em = -g.k_t * i_ts
    p_em = tau_em * omega_ts

    p_wind_mech = res.tau_wind * omega_ts
    eff = np.where(p_wind_mech > 1e-6, np.clip(p_load / p_wind_mech, 0.0, 1.2), 0.0)

    # Metrics
    m = compute_metrics(
        t=res.t,
        lambda_ts=res.lambda_ts,
        cp=res.cp,
        power_load=p_load,
        lambda_opt=mp.lambda_opt,
    )

    # Dataframe for export
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
        "p_copper": p_copper,
        "v_load": v_load,
        "p_wind_mech": p_wind_mech,
        "p_em": p_em,
        "eff_proxy": eff,
    })

    # -------------------------
    # DASHBOARD TAB
    # -------------------------
    with tab_dash:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Energy (Wh)", f"{m.energy_Wh:.2f}")
        c2.metric("Avg Cp", f"{m.avg_cp:.3f}")
        c3.metric("RMS λ error", f"{m.rms_lambda_error:.3f}")
        c4.metric("Settling time (s)", "N/A" if m.settling_time_s is None else f"{m.settling_time_s:.2f}")

        avg_power = float(np.mean(p_load))
        peak_power = float(np.max(p_load))
        avg_wind = float(np.mean(v_arr))
        max_cp = float(np.max(res.cp))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg Power (W)", f"{avg_power:.1f}")
        c6.metric("Peak Power (W)", f"{peak_power:.1f}")
        c7.metric("Avg Wind (m/s)", f"{avg_wind:.2f}")
        c8.metric("Max Cp", f"{max_cp:.3f}")

        fig_wp = go.Figure()
        fig_wp.add_trace(go.Scatter(x=res.t, y=v_arr, name="Wind (m/s)", mode="lines", yaxis="y1"))
        fig_wp.add_trace(go.Scatter(x=res.t, y=p_load, name="Power (W)", mode="lines", yaxis="y2"))
        fig_wp.update_layout(
            title="Wind & Power (Dual Axis)",
            xaxis_title="Time (s)",
            yaxis=dict(title="Wind (m/s)"),
            yaxis2=dict(title="Power (W)", overlaying="y", side="right"),
            template="plotly_white",
        )
        st.plotly_chart(fig_wp, use_container_width=True)

        fig_lambda = go.Figure()
        fig_lambda.add_trace(go.Scatter(x=res.t, y=res.lambda_ts, mode="lines", name="λ"))
        fig_lambda.add_trace(go.Scatter(x=res.t, y=[lambda_opt]*len(res.t), mode="lines",
                                        name="λ_opt", line=dict(dash="dash")))
        fig_lambda.update_layout(
            title="Tip-Speed Ratio λ",
            xaxis_title="Time (s)",
            yaxis_title="λ",
            template="plotly_white",
        )
        st.plotly_chart(fig_lambda, use_container_width=True)

    # -------------------------
    # ANALYSIS TAB
    # -------------------------
    with tab_analysis:
        fig_cplam = go.Figure()
        fig_cplam.add_trace(go.Scatter(
            x=res.lambda_ts, y=res.cp, mode="markers", name="Trajectory", marker=dict(size=5)
        ))
        fig_cplam.add_trace(go.Scatter(
            x=[lambda_opt, lambda_opt],
            y=[0, max(0.6, float(np.max(res.cp))*1.1)],
            mode="lines",
            name="λ_opt",
            line=dict(dash="dash"),
        ))
        fig_cplam.update_layout(
            title="Cp–λ Tracking (MPPT Performance)",
            xaxis_title="λ",
            yaxis_title="Cp",
            template="plotly_white",
        )
        st.plotly_chart(fig_cplam, use_container_width=True)

        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(x=res.t, y=p_wind_mech, mode="lines", name="P_wind_mech (W)"))
        fig_bal.add_trace(go.Scatter(x=res.t, y=p_em, mode="lines", name="P_em (W)"))
        fig_bal.add_trace(go.Scatter(x=res.t, y=p_load, mode="lines", name="P_load (W)"))
        fig_bal.add_trace(go.Scatter(x=res.t, y=p_copper, mode="lines", name="P_copper (W)"))
        fig_bal.update_layout(
            title="Power Balance",
            xaxis_title="Time (s)",
            yaxis_title="Power (W)",
            template="plotly_white",
        )
        st.plotly_chart(fig_bal, use_container_width=True)

        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scatter(x=res.t, y=eff, mode="lines", name="η proxy"))
        fig_eff.update_layout(
            title="Efficiency Proxy η = P_load / P_wind_mech",
            xaxis_title="Time (s)",
            yaxis_title="η",
            template="plotly_white",
        )
        st.plotly_chart(fig_eff, use_container_width=True)

        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=res.t, y=v_load, mode="lines", name="V_load (approx)"))
        fig_v.update_layout(
            title="Load Voltage (approx)",
            xaxis_title="Time (s)",
            yaxis_title="Voltage (V)",
            template="plotly_white",
        )
        st.plotly_chart(fig_v, use_container_width=True)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=res.t, y=res.R_load, mode="lines", name="R_load (Ω)"))
        fig_r.update_layout(
            title="MPPT Load Resistance",
            xaxis_title="Time (s)",
            yaxis_title="R_load (Ω)",
            template="plotly_white",
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # -------------------------
    # COMPARE TAB (save & overlay)
    # -------------------------
    with tab_compare:
        st.subheader("Save this run")
        if st.button("💾 Save this run"):
            st.session_state.runs[run_label] = {
                "t": res.t,
                "power": p_load,
                "lambda": res.lambda_ts,
                "cp": res.cp,
                "wind": v_arr,
            }
            st.success(f"Saved: {run_label}")

        st.divider()
        st.subheader("Compare saved runs")

        if len(st.session_state.runs) >= 2:
            selected = st.multiselect(
                "Select runs to compare (Power)",
                list(st.session_state.runs.keys()),
                default=list(st.session_state.runs.keys()),
            )

            fig_compare = go.Figure()
            for name in selected:
                data = st.session_state.runs[name]
                fig_compare.add_trace(go.Scatter(
                    x=data["t"], y=data["power"], mode="lines", name=name
                ))
            fig_compare.update_layout(
                title="Power Comparison",
                xaxis_title="Time (s)",
                yaxis_title="Power (W)",
                template="plotly_white",
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.info("Save at least 2 runs to compare.")

    # -------------------------
    # SWEEP TAB
    # -------------------------
    with tab_sweep:
        st.subheader("Power Curve Sweep")
        st.caption("Runs constant-wind simulations across speeds, plots average power.")

        v_min, v_max = st.slider("Wind range (m/s)", 2.0, 25.0, (3.0, 18.0), 0.5)
        n_pts = st.slider("Number of points", 5, 25, 12, 1)

        if st.button("Run Power Curve Sweep"):
            wind_speeds = np.linspace(v_min, v_max, n_pts)
            avg_powers = []

            with st.spinner("Running sweep..."):
                for v_test in wind_speeds:
                    test_wind = ConstantWind(v=float(v_test))
                    res_sweep = run_rotor_mppt_sim_profile(
                        wind=test_wind, p=p, g=g, a=a, mp=mp, cfg=cfg
                    )
                    avg_powers.append(float(np.mean(res_sweep.power_load)))

            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=wind_speeds,
                y=avg_powers,
                mode="lines+markers",
                name="Avg power",
            ))
            fig_curve.update_layout(
                title="Power Curve (MPPT)",
                xaxis_title="Wind speed (m/s)",
                yaxis_title="Average power (W)",
                template="plotly_white",
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            df_curve = pd.DataFrame({"v_wind": wind_speeds, "avg_power_W": avg_powers})
            st.download_button(
                "⬇ Download sweep CSV",
                data=df_curve.to_csv(index=False).encode("utf-8"),
                file_name="windforge_power_curve.csv",
                mime="text/csv",
            )

    # -------------------------
    # EXPORT TAB (ZIP bundle)
    # -------------------------
    with tab_export:
        st.subheader("Export")

        metrics_payload = {
            "energy_Wh": float(m.energy_Wh),
            "avg_cp": float(m.avg_cp),
            "rms_lambda_error": float(m.rms_lambda_error),
            "settling_time_s": None if m.settling_time_s is None else float(m.settling_time_s),
            "avg_power_W": float(np.mean(p_load)),
            "peak_power_W": float(np.max(p_load)),
        }

        # Create a main power figure for HTML export
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=res.t, y=p_load, mode="lines", name="Load Power (W)"))
        fig_power.update_layout(
            title="Electrical Power Output (W)",
            xaxis_title="Time (s)",
            yaxis_title="Power (W)",
            template="plotly_white",
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        metrics_bytes = json.dumps(metrics_payload, indent=2).encode("utf-8")
        html_bytes = fig_power.to_html(include_plotlyjs="cdn").encode("utf-8")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("windforge_results.csv", csv_bytes)
            z.writestr("metrics.json", metrics_bytes)
            z.writestr("report_power.html", html_bytes)
        buf.seek(0)

        st.download_button(
            "⬇ Download report bundle (ZIP)",
            data=buf.getvalue(),
            file_name="windforge_bundle.zip",
            mime="application/zip",
        )

        st.download_button(
            "⬇ Download results CSV",
            data=csv_bytes,
            file_name="windforge_results.csv",
            mime="text/csv",
        )

else:
    st.info("Set parameters in the sidebar, then click **Run simulation**.")