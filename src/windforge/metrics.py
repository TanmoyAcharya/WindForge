from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RunMetrics:
    energy_Wh: float
    avg_cp: float
    avg_lambda: float
    rms_lambda_error: float
    settling_time_s: float | None
    overshoot_lambda: float
    max_shaft_torque: float | None


def _trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def compute_metrics(
    t: np.ndarray,
    lambda_ts: np.ndarray,
    cp: np.ndarray,
    power_load: np.ndarray,
    lambda_opt: float,
    tau_shaft: np.ndarray | None = None,
    settling_band: float = 0.10,
    settle_window_s: float = 1.0,
) -> RunMetrics:
    """
    Compute standard wind-control metrics.

    settling_time_s:
      earliest time after which lambda stays within +/- (settling_band * lambda_opt)
      for settle_window_s seconds. Returns None if never settles.
    """
    t = np.asarray(t, dtype=float)
    lam = np.asarray(lambda_ts, dtype=float)
    cp = np.asarray(cp, dtype=float)
    p = np.asarray(power_load, dtype=float)

    # Energy in Wh
    energy_J = float(np.trapezoid(np.maximum(p, 0.0), t))
    energy_Wh = energy_J / 3600.0

    avg_cp = float(np.nanmean(cp))
    avg_lambda = float(np.nanmean(lam))
    rms_lambda_error = float(np.sqrt(np.nanmean((lam - lambda_opt) ** 2)))

    # Overshoot of lambda relative to lambda_opt
    overshoot_lambda = float(np.nanmax(lam) - lambda_opt)

    # Settling time
    band = abs(settling_band * lambda_opt)
    lo = lambda_opt - band
    hi = lambda_opt + band

    settling_time_s: float | None = None
    if len(t) >= 2:
        dt = float(np.median(np.diff(t)))
        win_n = max(1, int(round(settle_window_s / max(dt, 1e-9))))
        ok = (lam >= lo) & (lam <= hi)

        # Find earliest index i such that ok[i:i+win_n] are all True
        for i in range(0, len(t) - win_n):
            if bool(np.all(ok[i : i + win_n])):
                settling_time_s = float(t[i])
                break

    max_shaft_torque = None
    if tau_shaft is not None:
        ts = np.asarray(tau_shaft, dtype=float)
        max_shaft_torque = float(np.nanmax(np.abs(ts)))

    return RunMetrics(
        energy_Wh=energy_Wh,
        avg_cp=avg_cp,
        avg_lambda=avg_lambda,
        rms_lambda_error=rms_lambda_error,
        settling_time_s=settling_time_s,
        overshoot_lambda=overshoot_lambda,
        max_shaft_torque=max_shaft_torque,
    )