"""Generic joystick task used by multiple Codesign robots.

This is intentionally simpler than the legacy Hi joystick task: it only uses
IMU + joint state + foot contact for a robust bring-up baseline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import hi_base
from . import hi_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=1.0,
      tracking_sigma=0.25,
      noise_config=config_dict.create(
          level=0.0,
          scales=config_dict.create(
              joint_pos=0.0,
              joint_vel=0.0,
              gravity=0.0,
              linvel=0.0,
              gyro=0.0,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              lin_vel_z=-0.2,
              ang_vel_xy=-0.1,
              orientation=-1.0,
              action_rate=-0.01,
              feet_air_time=20.0,
              # feet_air_time=5.0,
              alive=0.5,
          ),
      ),
      push_config=config_dict.create(
          enable=False,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 1.0],
      ),
      # lin_vel_x=[-1.0, 1.0],
      # lin_vel_y=[-0.5, 0.5],
      # ang_vel_yaw=[-1.0, 1.0],

      lin_vel_x=[-0.5, 0.5],
      lin_vel_y=[-0.25, 0.25],
      ang_vel_yaw=[-1.0, 1.0],
  )


class BaseJoystick(hi_base.HiEnv):
  """Track a joystick command for a Codesign robot."""

  def __init__(
      self,
      *,
      xml_path: str,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(xml_path=xml_path, config=config, config_overrides=config_overrides)
    self._post_init()

  def _post_init(self) -> None:
    # Initialize from "home" keyframe if present, otherwise fall back to qpos0.
    try:
      keyframe = self._mj_model.keyframe("home")
      self._init_q = jp.array(keyframe.qpos)
      self._default_pose = jp.array(keyframe.qpos[7:])
    except KeyError:
      self._init_q = jp.array(self._mj_model.qpos0)
      self._default_pose = jp.array(self._mj_model.qpos0[7:])

    self._n_dof = int(self._mj_model.nq - 7)
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * 0.95
    self._soft_uppers = c + 0.5 * r * 0.95

    # Feet collision geoms + sites
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._left_foot_geom_id = self._mj_model.geom("left_foot").id
    self._right_foot_geom_id = self._mj_model.geom("right_foot").id
    self._left_feet_geom_id = [self._left_foot_geom_id]
    self._right_feet_geom_id = [self._right_foot_geom_id]
    self._feet_site_id = np.array([
        self._mj_model.site("left_foot").id,
        self._mj_model.site("right_foot").id,
    ])
    self._site_id = self._mj_model.site("imu").id

    # Used by rollout scripts for 2D animation. Our base body frame is (z up, y forward),
    # so yaw should be computed from the IMU site frame (x forward) when available.
    try:
      self._torso_body_id = int(self._mj_model.body(consts.ROOT_BODY).id)
    except Exception:  # pylint: disable=broad-except
      # Fallback: body 1 is typically the first real body (0 is world).
      self._torso_body_id = 1

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    vx = jax.random.uniform(k1, (1,), minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1])
    vy = jax.random.uniform(k2, (1,), minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1])
    wz = jax.random.uniform(k3, (1,), minval=self._config.ang_vel_yaw[0], maxval=self._config.ang_vel_yaw[1])
    return jp.hstack([vx, vy, wz])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Phase (gait clock) like HERMES NoLinearVel.
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.75)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0.0, jp.pi], dtype=jp.float32)

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=self._default_pose)

    left_contact = jp.any(jp.array([geoms_colliding(data, gid, self._floor_geom_id) for gid in self._left_feet_geom_id]))
    right_contact = jp.any(jp.array([geoms_colliding(data, gid, self._floor_geom_id) for gid in self._right_feet_geom_id]))
    contact = jp.hstack([left_contact, right_contact])

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        # Privileged-only signals.
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        # Phase (gait clock).
        "phase_dt": phase_dt,
        "phase": phase,
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Clamp targets inside soft joint range.
    motor_targets = jp.clip(self._default_pose + action * self._config.action_scale, self._soft_lowers, self._soft_uppers)
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

    left_contact = jp.any(jp.array([geoms_colliding(data, gid, self._floor_geom_id) for gid in self._left_feet_geom_id]))
    right_contact = jp.any(jp.array([geoms_colliding(data, gid, self._floor_geom_id) for gid in self._right_feet_geom_id]))
    contact = jp.hstack([left_contact, right_contact])

    # Feet air-time bookkeeping (privileged, like HERMES NoLinearVel).
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] = state.info["feet_air_time"] + self.dt

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, first_contact=first_contact, contact=contact)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    # Be robust to occasional numerical issues: never emit NaNs/Infs to PPO.
    total = jp.sum(jp.stack([jp.asarray(v) for v in rewards.values()]))
    total = jp.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    reward = jp.clip(total * self.dt, 0.0, 1e6)

    state.info["step"] += 1
    state.info["last_act"] = action
    # Reset air-time to 0 for feet that are in contact, like HERMES.
    state.info["feet_air_time"] = state.info["feet_air_time"] * (~contact)
    state.info["last_contact"] = contact

    # Update gait phase (HERMES-style). When command is ~0, hold phase at pi.
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["phase"] = jp.where(
        jp.linalg.norm(state.info["command"]) > 0.01,
        state.info["phase"],
        jp.ones(2, dtype=state.info["phase"].dtype) * jp.pi,
    )

    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    # Resample command periodically.
    state.info["command"] = jp.where(state.info["step"] > 500, self.sample_command(cmd_rng), state.info["command"])
    state.info["step"] = jp.where(done | (state.info["step"] > 500), 0, state.info["step"])

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    # If state ever becomes non-finite, terminate the episode.
    nonfinite = (~jp.all(jp.isfinite(data.qpos))) | (~jp.all(jp.isfinite(data.qvel)))
    done = (done | nonfinite).astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Fall if gravity points upward in body frame (robot upside-down).
    fall = self.get_gravity(data)[-1] < 0.0
    nonfinite = (~jp.all(jp.isfinite(data.qpos))) | (~jp.all(jp.isfinite(data.qvel)))
    return fall | nonfinite

  def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> dict[str, jax.Array]:
    # Actor observation ("state"): noisy IMU + joints, no linvel, no contact (NoLinearVel style).
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    # Base linear velocity (IMU-local). Actor does NOT see it (NoLinearVel),
    # but critic gets a noisy version via state_with_linvel (HERMES style).
    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    # Phase features (HERMES style): concat([cos(phase), sin(phase)]).
    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase_feat = jp.concatenate([cos, sin])

    state = jp.hstack([
        noisy_gyro,
        noisy_gravity,
        info["command"],
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
        phase_feat,
    ])

    # Critic observation ("privileged_state"): include a HERMES-style noisy prefix
    # (state_with_linvel) plus accurate + sim-only signals.
    state_with_linvel = jp.hstack([
        noisy_linvel,
        noisy_gyro,
        noisy_gravity,
        info["command"],
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
        phase_feat,
    ])

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    root_height = data.qpos[2]
    feet_vel = jp.hstack([
        mjx_env.get_sensor_data(self.mj_model, data, "left_foot_global_linvel").ravel(),
        mjx_env.get_sensor_data(self.mj_model, data, "right_foot_global_linvel").ravel(),
    ])
    privileged_state = jp.hstack([
        state_with_linvel,
        # Accurate proprioception.
        gyro,
        accelerometer,
        gravity,
        linvel,
        global_angvel,
        joint_angles - self._default_pose,
        joint_vel,
        # Sim-only signals.
        root_height[jp.newaxis],
        data.actuator_force,
        contact.astype(jp.float32),
        feet_vel,
        jp.asarray(info["feet_air_time"]),
    ])

    state = jp.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    state = jp.clip(state, -1e6, 1e6)
    privileged_state = jp.nan_to_num(privileged_state, nan=0.0, posinf=0.0, neginf=0.0)
    privileged_state = jp.clip(privileged_state, -1e6, 1e6)

    return {"state": state, "privileged_state": privileged_state}

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    air = (air_time - threshold_min) * first_contact
    air = jp.clip(air, a_min=0.0, a_max=threshold_max - threshold_min)
    reward = jp.sum(air)
    return reward * (cmd_norm > 0.01)

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      *,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    cmd = info["command"]
    # Track commands in the IMU-local convention (x forward, z up), same as HERMES.
    local_linvel = self.get_local_linvel(data)
    local_angvel = self.get_gyro(data)
    local_linvel = jp.nan_to_num(local_linvel, nan=0.0, posinf=0.0, neginf=0.0)
    local_angvel = jp.nan_to_num(local_angvel, nan=0.0, posinf=0.0, neginf=0.0)

    vel_err = jp.sum(jp.square(local_linvel[:2] - cmd[:2]))
    yaw_err = jp.square(local_angvel[2] - cmd[2])
    tracking_lin = jp.exp(-vel_err / self._config.tracking_sigma)
    tracking_yaw = jp.exp(-yaw_err / self._config.tracking_sigma)

    gravity = self.get_gravity(data)
    orientation = jp.sum(jp.square(gravity[:2]))

    return {
        "tracking_lin_vel": tracking_lin,
        "tracking_ang_vel": tracking_yaw,
        "lin_vel_z": jp.square(local_linvel[2]),
        "ang_vel_xy": jp.sum(jp.square(local_angvel[:2])),
        "orientation": orientation,
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "feet_air_time": self._reward_feet_air_time(info["feet_air_time"], first_contact, cmd),
        "alive": jp.array(1.0),
    }


