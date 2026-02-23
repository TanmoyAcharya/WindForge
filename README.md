# WindForge

WindForge is a lightweight scientific computing library for simulating rotor dynamics in wind energy systems.

It converts mechanical engineering models into reusable Python simulation tools for time-domain numerical analysis, parametric studies, and visualization.

---

## Overview

WindForge models the rotational dynamics of a wind-driven rotor with:

- Aerodynamic torque model
- Load torque model
- Viscous and Coulomb friction
- Numerical ODE integration (SciPy)
- Parametric wind-speed sweeps
- Automated steady-state power analysis
- Visualization tools

This project demonstrates how mechanical system equations can be transformed into structured, reusable Python software.

---

## Governing Equation

The rotor dynamics are modeled using:

\[
I \dot{\omega} = \tau_{wind} - \tau_{load} - b\omega - \tau_c \, sign(\omega)
\]

Where:

- \( I \) — Rotor inertia
- \( \omega \) — Angular velocity
- \( b \) — Viscous damping coefficient
- \( \tau_c \) — Coulomb friction torque
- \( \tau_{wind} \) — Aerodynamic torque
- \( \tau_{load} \) — Load torque (e.g., generator)

The system is solved using `scipy.integrate.solve_ivp`.

---

## Project Structure
WindForge/
src/windforge/
rotor.py # Mechanical model
sim.py # ODE solver and simulation logic
plots.py # Visualization tools
examples/
run_single_case.py
run_wind_sweep.py



---

## Installation

```bash
git clone https://github.com/TanmoyAcharya/WindForge.git
cd WindForge
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Quick start

Activate venv:

```powershell
.\.venv\Scripts\Activate
$env:PYTHONPATH=".\src"