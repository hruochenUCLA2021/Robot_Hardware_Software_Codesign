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
      action_scale=0.5,
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
              alive=0.5,
          ),
      ),
      push_config=config_dict.create(
          enable=False,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 1.0],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-0.5, 0.5],
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

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    vx = jax.random.uniform(k1, (1,), minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1])
    vy = jax.random.uniform(k2, (1,), minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1])
    wz = jax.random.uniform(k3, (1,), minval=self._config.ang_vel_yaw[0], maxval=self._config.ang_vel_yaw[1])
    return jp.hstack([vx, vy, wz])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

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

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    # Be robust to occasional numerical issues: never emit NaNs/Infs to PPO.
    total = jp.sum(jp.stack([jp.asarray(v) for v in rewards.values()]))
    total = jp.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    reward = jp.clip(total * self.dt, 0.0, 1e6)

    state.info["step"] += 1
    state.info["last_act"] = action
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
    gyro = self.get_gyro(data)
    linvel = self.get_local_linvel(data)
    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]

    state = jp.hstack([
        linvel,
        gyro,
        gravity,
        info["command"],
        joint_angles - self._default_pose,
        joint_vel,
        info["last_act"],
        contact.astype(jp.float32),
    ])
    state = jp.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    state = jp.clip(state, -1e6, 1e6)

    return {
        "state": state
    }

  def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict[str, Any]) -> dict[str, jax.Array]:
    cmd = info["command"]
    # Use world-frame sensors for tracking.
    glin = self.get_global_linvel(data)
    gang = self.get_global_angvel(data)
    glin = jp.nan_to_num(glin, nan=0.0, posinf=0.0, neginf=0.0)
    gang = jp.nan_to_num(gang, nan=0.0, posinf=0.0, neginf=0.0)

    vel_err = jp.sum(jp.square(glin[:2] - cmd[:2]))
    yaw_err = jp.square(gang[2] - cmd[2])
    tracking_lin = jp.exp(-vel_err / self._config.tracking_sigma)
    tracking_yaw = jp.exp(-yaw_err / self._config.tracking_sigma)

    gravity = self.get_gravity(data)
    orientation = jp.sum(jp.square(gravity[:2]))

    return {
        "tracking_lin_vel": tracking_lin,
        "tracking_ang_vel": tracking_yaw,
        "lin_vel_z": jp.square(glin[2]),
        "ang_vel_xy": jp.sum(jp.square(gang[:2])),
        "orientation": orientation,
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "alive": jp.array(1.0),
    }


