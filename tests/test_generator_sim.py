import numpy as np

from windforge.rotor import RotorParams
from windforge.generator import GeneratorParams
from windforge.sim import SimConfig, run_rotor_gen_sim


def steady_slice(x: np.ndarray, frac: float = 0.8) -> np.ndarray:
    n = len(x)
    i0 = int(frac * n)
    return x[i0:]


def test_sim_runs_and_shapes():
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1, k_wind=0.6)
    g = GeneratorParams(R_g=0.6, L=0.02, k_e=0.25, k_t=0.25, R_load=2.0)
    cfg = SimConfig(t_end=2.0, dt=0.01)  # short test

    res = run_rotor_gen_sim(v_wind=8.0, p=p, g=g, cfg=cfg)

    assert res.t.ndim == 1
    assert res.omega.shape == res.t.shape
    assert res.i.shape == res.t.shape
    assert res.power_load.shape == res.t.shape
    assert np.isfinite(res.omega).all()
    assert np.isfinite(res.i).all()


def test_power_balance_in_steady_state_is_reasonable():
    """
    In steady state, mechanical->electrical converted power should roughly match
    load power + copper loss (up to modeling simplifications and transients).
    """
    p = RotorParams(I=0.25, b=0.02, tau_c=0.1, k_wind=0.6)
    g = GeneratorParams(R_g=0.6, L=0.02, k_e=0.25, k_t=0.25, R_load=2.0)
    cfg = SimConfig(t_end=10.0, dt=0.01)

    res = run_rotor_gen_sim(v_wind=8.0, p=p, g=g, cfg=cfg)

    pem = float(np.mean(steady_slice(res.power_em)))
    pel = float(np.mean(steady_slice(res.power_load + res.power_copper)))

    # allow 30% relative error due to simple model + friction + non-ideal steady window
    if pem > 1e-6:
        rel_err = abs(pem - pel) / pem
        assert rel_err < 0.30