from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import math


@runtime_checkable
class WindProfile(Protocol):
    def __call__(self, t: float) -> float: ...


@dataclass(frozen=True)
class ConstantWind:
    v: float = 8.0
    def __call__(self, t: float) -> float:
        return float(self.v)


@dataclass(frozen=True)
class StepGust:
    v0: float = 7.0
    v1: float = 11.0
    t_step: float = 8.0
    def __call__(self, t: float) -> float:
        return float(self.v1 if t >= self.t_step else self.v0)


@dataclass(frozen=True)
class RampWind:
    v0: float = 6.0
    v1: float = 12.0
    t0: float = 2.0
    t1: float = 12.0
    def __call__(self, t: float) -> float:
        if t <= self.t0:
            return float(self.v0)
        if t >= self.t1:
            return float(self.v1)
        alpha = (t - self.t0) / (self.t1 - self.t0)
        return float(self.v0 + alpha * (self.v1 - self.v0))


@dataclass(frozen=True)
class SineWind:
    v_mean: float = 9.0
    amp: float = 2.0
    freq_hz: float = 0.08
    def __call__(self, t: float) -> float:
        return float(self.v_mean + self.amp * math.sin(2.0 * math.pi * self.freq_hz * t))