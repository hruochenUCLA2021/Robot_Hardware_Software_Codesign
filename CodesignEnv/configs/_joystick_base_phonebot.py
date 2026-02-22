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
          level=0.4,
          scales=config_dict.create(
              # joint_pos=0.0,
              # joint_vel=0.0,
              # gravity=0.0,
              # linvel=0.0,
              # gyro=0.0,
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              lin_vel_z=-0.2,
              ang_vel_xy=-0.1,
              orientation=-1.0,
              # Energy-related penalties (HERMES style). Keep 0 by default; tune later.
              torques=0.0,
              energy=0.0,
              action_rate=-0.01,
              # dof_acc=0.0,
              # dof_acc=-0.00000001,
              dof_acc=-0.000004,
              # dof_vel=0.0,
              # dof_vel=-0.00001,
              dof_vel=-0.0001,
              feet_air_time=20.0,
              # Foot-related costs (tune later).
              feet_clearance=-0.05,
              foot_collision=-1.0,
              # Pose related rewards (HERMES NoLinearVel style).
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.1,
              dof_pos_limits=-1.0,
              pose=-0.25,
              # pose=-1.0,
              feet_distance=-1.0,
              # feet_air_time=50.0,
              # HERMES-style swing-peak based foot height penalty.
              # Keep at 0.0 for now; you can turn it on later if feet skim the ground.
              feet_height=-5.0, # it should be negative , since it is penalty !!!!! 
              
              # feet_air_time=20.0,
              # feet_air_time=5.0,
              alive=0.5,
          ),
          # Reference swing peak height (meters) for foot-height cost.
          max_foot_height=0.06,
          # max_foot_height=0.20,
      ),
      push_config=config_dict.create(
          enable=True,
          # enable=False,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.05, 0.2], # add small push to the robot
          # magnitude_range=[0.1, 1.0],
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

    # Pose-related indices + weights (HERMES NoLinearVel style).
    # Hip indices: use hip roll joints.
    self._hip_indices = jp.array(
        [
            self._mj_model.joint("l_hip_roll_joint").qposadr - 7,
            self._mj_model.joint("r_hip_roll_joint").qposadr - 7,
        ],
        dtype=jp.int32,
    )
    # Knee indices: use calf joints.
    self._knee_indices = jp.array(
        [
            self._mj_model.joint("l_hip_calf_joint").qposadr - 7,
            self._mj_model.joint("r_hip_calf_joint").qposadr - 7,
        ],
        dtype=jp.int32,
    )
    # Weights for pose cost: equal weights for all DOFs.
    self._weights = jp.ones((self._n_dof,), dtype=jp.float32)

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

    # Sample push interval (HERMES-style).
    rng, push_rng = jax.random.split(rng)
    push_cfg = self._config.push_config
    push_interval = jax.random.uniform(
        push_rng,
        minval=push_cfg.interval_range[0],
        maxval=push_cfg.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

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
        # Swing peak (used by foot-height cost; HERMES NoLinearVel style).
        "swing_peak": jp.zeros(2),
        # Phase (gait clock).
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0], dtype=jp.float32),
        "push_step": jp.asarray(0, dtype=jp.int32),
        "push_interval_steps": push_interval_steps,
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    metrics["swing_peak"] = jp.zeros(())
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Push disturbance (HERMES-style): inject small random XY velocity at intervals.
    state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)], dtype=jp.float32)
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
    )
    push *= self._config.push_config.enable
    state.info["push"] = push

    qvel0 = state.data.qvel
    qvel0 = qvel0.at[:2].set(push * push_magnitude + qvel0[:2])
    data0 = state.data.replace(qvel=qvel0)
    state = state.replace(data=data0)

    # Clamp targets inside soft joint range.
    motor_targets = jp.clip(
        self._default_pose + action * self._config.action_scale,
        self._soft_lowers,
        self._soft_uppers,
    )
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

    left_contact = jp.any(jp.array([geoms_colliding(data, gid, self._floor_geom_id) for gid in self._left_feet_geom_id]))
    right_contact = jp.any(jp.array([geoms_colliding(data, gid, self._floor_geom_id) for gid in self._right_feet_geom_id]))
    contact = jp.hstack([left_contact, right_contact])

    # Feet air-time bookkeeping (privileged, like HERMES NoLinearVel).
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] = state.info["feet_air_time"] + self.dt

    # Track swing peak (max foot height since last contact), HERMES-style.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], foot_z)

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, first_contact=first_contact, contact=contact)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    # Be robust to occasional numerical issues: never emit NaNs/Infs to PPO.
    total = jp.sum(jp.stack([jp.asarray(v) for v in rewards.values()]))
    total = jp.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    reward = jp.clip(total * self.dt, 0.0, 1e6)

    state.info["step"] += 1
    state.info["push_step"] += 1
    state.info["last_act"] = action
    # Reset air-time to 0 for feet that are in contact, like HERMES.
    state.info["feet_air_time"] = state.info["feet_air_time"] * (~contact)
    state.info["last_contact"] = contact
    state.info["swing_peak"] = state.info["swing_peak"] * (~contact)

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
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

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

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
  ) -> jax.Array:
    # HERMES NoLinearVel: penalize deviation of swing peak from a target height,
    # applied only on first contact.
    first_contact = jp.asarray(first_contact, dtype=jp.float32)
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  # Energy related costs (HERMES NoLinearVel style).
  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(self, qvel: jax.Array, actuator_force: jax.Array) -> jax.Array:
    # IMPORTANT: must be a Python int for slicing under `jax.jit`.
    n = min(qvel.shape[0], actuator_force.shape[0])
    return jp.sum(jp.abs(qvel[:n] * actuator_force[:n]))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qvel))

  # Pose-related costs (HERMES NoLinearVel style).
  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_joint_deviation_hip(self, qpos: jax.Array, cmd: jax.Array) -> jax.Array:
    cost = jp.sum(jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices]))
    cost *= jp.abs(cmd[1]) > 0.1
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(qpos[self._knee_indices] - self._default_pose[self._knee_indices]))

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  def _cost_feet_distance(self, data: mjx.Data) -> jax.Array:
    left_foot_pos = data.site_xpos[self._feet_site_id[0]]
    right_foot_pos = data.site_xpos[self._feet_site_id[1]]
    base_xmat = data.site_xmat[self._site_id]
    base_yaw = jp.arctan2(base_xmat[1, 0], base_xmat[0, 0])
    feet_distance = jp.abs(
        jp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1])
        - jp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
    )
    return jp.clip(0.2 - feet_distance, 0.0, 0.1)

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    # HERMES-style: penalize deviation from target foot height during swing,
    # weighted by horizontal foot speed.
    feet_vel = jp.stack(
        [
            mjx_env.get_sensor_data(self.mj_model, data, "left_foot_global_linvel").ravel(),
            mjx_env.get_sensor_data(self.mj_model, data, "right_foot_global_linvel").ravel(),
        ],
        axis=0,
    )
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_foot_collision(self, data: mjx.Data) -> jax.Array:
    # Foot tip vs foot tip collision (proxy for self-collision).
    return jp.asarray(
        geoms_colliding(data, self._left_foot_geom_id, self._right_foot_geom_id),
        dtype=jp.float32,
    )

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

    # Energy-related costs (non-root DOFs).
    joint_qvel = data.qvel[6:]
    joint_qacc = getattr(data, "qacc", None)
    joint_qacc = jp.zeros_like(joint_qvel) if joint_qacc is None else joint_qacc[6:]
    torques_cost = self._cost_torques(data.actuator_force)
    energy_cost = self._cost_energy(joint_qvel, data.actuator_force)
    dof_acc_cost = self._cost_dof_acc(joint_qacc)
    dof_vel_cost = self._cost_dof_vel(joint_qvel)
    feet_clearance_cost = self._cost_feet_clearance(data)
    foot_collision_cost = self._cost_foot_collision(data)
    # Pose-related costs.
    qpos_j = data.qpos[7:]
    joint_deviation_hip_cost = self._cost_joint_deviation_hip(qpos_j, cmd)
    joint_deviation_knee_cost = self._cost_joint_deviation_knee(qpos_j)
    joint_pos_limits_cost = self._cost_joint_pos_limits(qpos_j)
    pose_cost = self._cost_pose(qpos_j)
    feet_distance_cost = self._cost_feet_distance(data)

    return {
        "tracking_lin_vel": tracking_lin,
        "tracking_ang_vel": tracking_yaw,
        "lin_vel_z": jp.square(local_linvel[2]),
        "ang_vel_xy": jp.sum(jp.square(local_angvel[:2])),
        "orientation": orientation,
        "torques": torques_cost,
        "energy": energy_cost,
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "dof_acc": dof_acc_cost,
        "dof_vel": dof_vel_cost,
        "feet_air_time": self._reward_feet_air_time(info["feet_air_time"], first_contact, cmd),
        "feet_clearance": feet_clearance_cost,
        "foot_collision": foot_collision_cost,
        "joint_deviation_knee": joint_deviation_knee_cost,
        "joint_deviation_hip": joint_deviation_hip_cost,
        "dof_pos_limits": joint_pos_limits_cost,
        "pose": pose_cost,
        "feet_distance": feet_distance_cost,
        "feet_height": self._cost_feet_height(info["swing_peak"], first_contact),
        "alive": jp.array(1.0),
    }


