"""Motor control utilities for CodesignEnv.

This module implements a ToddlerBot-style torque-limited PD controller:
- Policy outputs joint position targets.
- Controller converts targets into torque commands.
- Torque is saturated by a velocity-dependent (piecewise linear) envelope with
  an asymmetric constant braking limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jp


@dataclass(frozen=True)
class TorqueLimitedPDParams:
  """Parameters for a torque-limited PD motor model (per-DOF)."""

  kp: jp.ndarray
  kd: jp.ndarray
  kd_min: jp.ndarray
  passive_active_ratio: float

  # Acceleration-side torque-speed envelope
  tau_max: jp.ndarray
  q_dot_tau_max: jp.ndarray
  q_dot_max: jp.ndarray
  tau_q_dot_max: jp.ndarray

  # Brake-side limit (constant)
  tau_brake_max: jp.ndarray


class TorqueLimitedPDController:
  """Convert position targets into torque commands with velocity-dependent limits."""

  def __init__(self, params: TorqueLimitedPDParams):
    self._p = params

  def step(
      self,
      q: jp.ndarray,
      q_dot: jp.ndarray,
      q_dot_dot: jp.ndarray,
      q_target: jp.ndarray,
      noise: Mapping[str, jp.ndarray] | None = None,
  ) -> jp.ndarray:
    """Compute saturated motor torque command.

    Args:
      q: joint position (rad), shape (n,)
      q_dot: joint velocity (rad/s), shape (n,)
      q_dot_dot: joint acceleration (rad/s^2), shape (n,)
      q_target: desired joint position target (rad), shape (n,)
      noise: optional multiplicative noise dict (keys match param names)

    Returns:
      Torque command (Nm), shape (n,).
    """
    n = noise or {}
    kp = self._p.kp * n.get("kp", 1.0)
    kd = self._p.kd * n.get("kd", 1.0)
    kd_min = self._p.kd_min * n.get("kd_min", 1.0)
    passive_active_ratio = self._p.passive_active_ratio * float(n.get("passive_active_ratio", 1.0))

    tau_max = self._p.tau_max * n.get("tau_max", 1.0)
    q_dot_tau_max = self._p.q_dot_tau_max * n.get("q_dot_tau_max", 1.0)
    q_dot_max = self._p.q_dot_max * n.get("q_dot_max", 1.0)
    tau_q_dot_max = self._p.tau_q_dot_max * n.get("tau_q_dot_max", 1.0)
    tau_brake_max = self._p.tau_brake_max * n.get("tau_brake_max", 1.0)

    error = q_target - q
    # Backdrive compensation (ToddlerBot-style): increase kp when acceleration opposes error.
    real_kp = jp.where(q_dot_dot * error < 0.0, kp * passive_active_ratio, kp)
    tau_m = real_kp * error - (kd_min + kd) * q_dot

    abs_q_dot = jp.abs(q_dot)

    # Piecewise-linear acceleration-side limit:
    # - tau_max for |q_dot| <= q_dot_tau_max
    # - linear taper to tau_q_dot_max at |q_dot| = q_dot_max
    # - keep taper beyond q_dot_max (we still clamp by brake/acc bounds below)
    denom = jp.maximum(q_dot_max - q_dot_tau_max, 1e-6)
    slope = (tau_q_dot_max - tau_max) / denom
    taper_limit = tau_max + slope * (abs_q_dot - q_dot_tau_max)
    tau_acc_limit = jp.where(abs_q_dot <= q_dot_tau_max, tau_max, taper_limit)

    # Asymmetric clamp: allow larger magnitude on the braking side.
    tau_clamped_pos = jp.clip(tau_m, -tau_brake_max, tau_acc_limit)   # q_dot >= 0
    tau_clamped_neg = jp.clip(tau_m, -tau_acc_limit, tau_brake_max)   # q_dot < 0
    tau_m_clamped = jp.where(q_dot >= 0.0, tau_clamped_pos, tau_clamped_neg)

    # Optional "self-protection": if exceeding max speed and trying to accelerate further,
    # apply braking torque (ToddlerBot-style).
    accelerating = (q_dot * error) > 0.0
    overspeed = abs_q_dot > q_dot_max
    protect = overspeed & accelerating
    protect_tau = jp.where(q_dot > 0.0, -tau_brake_max, tau_brake_max)
    return jp.where(protect, protect_tau, tau_m_clamped)

