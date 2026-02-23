from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import RunMetrics


def ensure_outputs_dir(out_dir: str | Path = "outputs") -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_timeseries_csv(out_dir: str | Path, name: str, data: dict) -> Path:
    out = ensure_outputs_dir(out_dir) / f"{name}.csv"
    df = pd.DataFrame(data)
    df.to_csv(out, index=False)
    return out


def save_metrics_json(out_dir: str | Path, name: str, m: RunMetrics) -> Path:
    out = ensure_outputs_dir(out_dir) / f"{name}_metrics.json"
    out.write_text(json.dumps(asdict(m), indent=2), encoding="utf-8")
    return out


def save_summary_plots(
    out_dir: str | Path,
    name: str,
    t: np.ndarray,
    v_wind: np.ndarray,
    lambda_ts: np.ndarray,
    cp: np.ndarray,
    r_load: np.ndarray,
    power_load: np.ndarray,
    omega_r: np.ndarray | None = None,
    omega_g: np.ndarray | None = None,
    omega_ref: np.ndarray | None = None,
    tau_shaft: np.ndarray | None = None,
    lambda_opt: float | None = None,
) -> Path:
    """
    Save a single multi-panel PNG that looks good in README / reports.
    """
    out_dir = ensure_outputs_dir(out_dir)

    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(t, v_wind)
    ax1.set_title("Wind speed (m/s)")
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, lambda_ts, label="lambda")
    if lambda_opt is not None:
        ax2.axhline(lambda_opt, linestyle="--", linewidth=1, label="lambda_opt")
    ax2.set_title("Tip-speed ratio λ")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, cp)
    ax3.set_title("Power coefficient Cp")
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, r_load)
    ax4.set_title("MPPT load command R_load (Ω)")
    ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, power_load)
    ax5.set_title("Electrical load power (W)")
    ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)
    if omega_r is not None:
        ax6.plot(t, omega_r, label="omega_r")
    if omega_g is not None:
        ax6.plot(t, omega_g, label="omega_g")
    if omega_ref is not None:
        ax6.plot(t, omega_ref, "--", label="omega_ref")
    if tau_shaft is not None:
        ax6b = ax6.twinx()
        ax6b.plot(t, tau_shaft, linestyle=":", linewidth=1, label="tau_shaft")
        ax6b.set_ylabel("Shaft torque (N·m)")
    ax6.set_title("Speeds / torsion")
    ax6.grid(True)
    ax6.legend(loc="upper left")

    fig.tight_layout()
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def write_report_md(
    out_dir: str | Path,
    name: str,
    title: str,
    description: str,
    metrics: RunMetrics,
    plot_path: Path,
    csv_path: Path,
) -> Path:
    out_dir = ensure_outputs_dir(out_dir)
    out = out_dir / f"{name}_report.md"

    md = []
    md.append(f"# {title}\n")
    md.append(description.strip() + "\n")
    md.append("## Key metrics\n")
    md.append(f"- Energy captured: **{metrics.energy_Wh:.2f} Wh**")
    md.append(f"- Avg Cp: **{metrics.avg_cp:.3f}**")
    md.append(f"- Avg λ: **{metrics.avg_lambda:.3f}**")
    md.append(f"- RMS(λ − λ_opt): **{metrics.rms_lambda_error:.3f}**")
    md.append(f"- λ overshoot: **{metrics.overshoot_lambda:.3f}**")
    if metrics.settling_time_s is None:
        md.append("- Settling time: **did not settle**")
    else:
        md.append(f"- Settling time: **{metrics.settling_time_s:.2f} s**")
    if metrics.max_shaft_torque is not None:
        md.append(f"- Max |shaft torque|: **{metrics.max_shaft_torque:.2f} N·m**")

    md.append("\n## Outputs\n")
    md.append(f"- Plot: `{plot_path.name}`")
    md.append(f"- Timeseries: `{csv_path.name}`")
    md.append("\n## Plot\n")
    md.append(f"![plot]({plot_path.name})\n")

    out.write_text("\n".join(md), encoding="utf-8")
    return out