# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for CodesignEnv (PhoneBot + ChopstickBot).

This is a lightweight, robot-agnostic port of the HERMES/Playground randomizers.

Key properties:
- **Adaptive**: no hardcoded joint counts. Uses `model.nu` (actuators) and
  randomizes the corresponding robot DOFs at indices `[6 : 6 + nu]`.
- **Multiplicative scaling**: for small robots, we prefer multiplying by a scale
  (e.g. mass *= U(0.95, 1.05)) rather than adding an absolute delta.
- **Safe by default**: only touches dynamics parameters that are known to be
  compatible with MJX/Brax wrappers.
"""

# import jax
# from mujoco import mjx

# from mujoco_playground._src.locomotion.t1 import randomize as t1_randomize


# def domain_randomize(model: mjx.Model, rng: jax.Array):
#   """Delegate to the official T1 domain randomizer (JAX-based).

#   Args:
#     model: mjx.Model for the Hi robot.
#     rng: JAX PRNG key or batch of keys, supplied by Brax.

#   Returns:
#     A tuple (model_v, in_axes) matching the expected contract of
#     `wrapper.wrap_for_brax_training` and Brax's PPO.
#   """
#   return t1_randomize.domain_randomize(model, rng)

import jax
import jax.numpy as jp
from mujoco import mjx


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Randomize MJX model parameters.

  Args:
    model: `mjx.Model` (single model).
    rng: JAX PRNG key(s). Brax typically provides a batch of keys and expects
      `domain_randomize` to be vmapped over it.

  Returns:
    (model_v, in_axes) where `model_v` has a leading batch dimension on the
    randomized leaves, and `in_axes` marks which leaves are batched (0) vs
    static (None). This matches Brax/MuJoCo Playground conventions.
  """

  nu = int(model.nu)
  # Robot DOFs are the first `nu` hinge DOFs after the 6-DOF free base.
  robot_dof_slice = slice(6, 6 + nu)
  # Robot qpos0 are the joint coordinates after the 7 free-joint coordinates.
  robot_qpos_slice = slice(7, 7 + nu)

  def _u(rng_key, *, shape=(), lo=0.0, hi=1.0):
    return jax.random.uniform(rng_key, shape=shape, minval=lo, maxval=hi)

  @jax.vmap
  def rand_dynamics(key):
    # Split keys in a fixed order for reproducibility.
    k_fric, k_floss, k_arm, k_mass, k_qpos0, k_kp, k_kd = jax.random.split(key, 7)

    # -----------------------------------------------------------------------
    # Contact friction (scale all geoms uniformly).
    # geom_friction: (ngeom, 3) = [sliding, torsional, rolling].
    # -----------------------------------------------------------------------
    fric_scale = _u(k_fric, lo=0.7, hi=1.3)
    geom_friction = model.geom_friction * fric_scale

    # -----------------------------------------------------------------------
    # DOF frictionloss / armature / damping (robot DOFs only).
    # -----------------------------------------------------------------------
    if nu > 0:
      fl_scale = _u(k_floss, shape=(nu,), lo=0.7, hi=1.3)
      ar_scale = _u(k_arm, shape=(nu,), lo=0.8, hi=1.2)
      kd_scale = _u(k_kd, shape=(nu,), lo=0.7, hi=1.3)

      dof_frictionloss = model.dof_frictionloss.at[robot_dof_slice].set(
          model.dof_frictionloss[robot_dof_slice] * fl_scale
      )
      dof_armature = model.dof_armature.at[robot_dof_slice].set(
          model.dof_armature[robot_dof_slice] * ar_scale
      )
      dof_damping = model.dof_damping.at[robot_dof_slice].set(
          model.dof_damping[robot_dof_slice] * kd_scale
      )
    else:
      dof_frictionloss = model.dof_frictionloss
      dof_armature = model.dof_armature
      dof_damping = model.dof_damping

    # -----------------------------------------------------------------------
    # Body masses (scale per-body).
    # -----------------------------------------------------------------------
    body_mass_scale = _u(k_mass, shape=(int(model.nbody),), lo=0.95, hi=1.05)
    body_mass = model.body_mass * body_mass_scale

    # -----------------------------------------------------------------------
    # Jitter robot qpos0 only (keeps freejoint quaternion valid).
    # -----------------------------------------------------------------------
    if nu > 0:
      dq = _u(k_qpos0, shape=(nu,), lo=-0.05, hi=0.05)
      qpos0 = model.qpos0.at[robot_qpos_slice].set(model.qpos0[robot_qpos_slice] + dq)
    else:
      qpos0 = model.qpos0

    # -----------------------------------------------------------------------
    # Actuator position gains (kp): scale actuator_gainprm[:, 0].
    # Also mirror into biasprm[:, 1] = -kp (Playground convention).
    # -----------------------------------------------------------------------
    kp_scale = _u(k_kp, shape=(int(model.nu),), lo=0.8, hi=1.2) if int(model.nu) > 0 else jp.zeros((0,))
    actuator_gainprm = model.actuator_gainprm
    actuator_biasprm = model.actuator_biasprm
    if int(model.nu) > 0:
      kp = model.actuator_gainprm[:, 0] * kp_scale
      actuator_gainprm = actuator_gainprm.at[:, 0].set(kp)
      actuator_biasprm = actuator_biasprm.at[:, 1].set(-kp)

    return (
        geom_friction,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
    )

  (
      geom_friction,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      body_mass,
      qpos0,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand_dynamics(rng)

  # Mark which leaves are batched (leading axis 0).
  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  in_axes = in_axes.tree_replace(
      {
          "geom_friction": 0,
          "dof_frictionloss": 0,
          "dof_armature": 0,
          "dof_damping": 0,
          "body_mass": 0,
          "qpos0": 0,
          "actuator_gainprm": 0,
          "actuator_biasprm": 0,
      }
  )

  model_v = model.tree_replace(
      {
          "geom_friction": geom_friction,
          "dof_frictionloss": dof_frictionloss,
          "dof_armature": dof_armature,
          "dof_damping": dof_damping,
          "body_mass": body_mass,
          "qpos0": qpos0,
          "actuator_gainprm": actuator_gainprm,
          "actuator_biasprm": actuator_biasprm,
      }
  )
  return model_v, in_axes
