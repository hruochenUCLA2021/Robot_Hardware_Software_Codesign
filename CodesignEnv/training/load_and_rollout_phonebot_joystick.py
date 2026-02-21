#!/usr/bin/env python3
"""
Load and rollout PhonebotJoystick* policy.

This script:
- Loads a trained Phonebot joystick policy from a checkpoint directory.
- Runs a rollout for up to 3000 steps (or until `state.done` is True).
- Prints basic diagnostics during rollout.
"""

import os
import sys
import functools
from datetime import datetime

# ---------------------------------------------------------------------------
# Environment / import setup
# ---------------------------------------------------------------------------

# Configure Mujoco/OpenGL and JAX memory behaviour *before* importing anything
# that might indirectly import `mujoco` or JAX.
# Avoid preallocating all GPU memory up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Try to use an offscreen GL backend (EGL) so rendering works over SSH/headless.
# If your Mujoco build does not support EGL, you can instead try "osmesa" or
# unset this variable to use the default GLFW-based backend on a local desktop.
os.environ.setdefault("MUJOCO_GL", "egl")

# Enable JAX persistent compilation cache to speed up repeated rollouts on this
# machine. These match the training script settings so that compiled executables
# can be reused across runs.
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

import jax
import jax.numpy as jp
import numpy as np
from etils import epath
from ml_collections import config_dict
import mediapy as media
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mujoco
import yaml

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

# Ensure project root (containing Robot_Hardware_Software_Codesign) is on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# training/ -> CodesignEnv/ -> Robot_Hardware_Software_Codesign/
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

# Import local CodesignEnv registry.
from CodesignEnv import registry as env_registry  # noqa: E402


# ---------------------------------------------------------------------------
# PPO config (reuse T1 config as base)
# ---------------------------------------------------------------------------

def get_ppo_params_for_hi() -> config_dict.ConfigDict:
  """Get PPO hyperparameters, using T1 flat terrain as a reference."""
  try:
    ppo_params = locomotion_params.brax_ppo_config("T1JoystickFlatTerrain")
    # CodesignEnv joystick env exposes obs dict with keys:
    # - policy (actor): "state"
    # - value  (critic): "privileged_state"
    if "network_factory" in ppo_params:
      ppo_params["network_factory"]["policy_obs_key"] = "state"
      ppo_params["network_factory"]["value_obs_key"] = "privileged_state"
  except Exception as e:  # pylint: disable=broad-except
    print(f"Warning: failed to load T1 PPO params ({e}), using defaults.")
    ppo_params = config_dict.ConfigDict(
        {
            "num_timesteps": 5_000_000,
            "num_evals": 50,
            "reward_scaling": 1.0,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 10,
            "num_minibatches": 16,
            "num_updates_per_batch": 4,
            "discounting": 0.99,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-4,
            "num_envs": 1,
            "batch_size": 1,
            "num_eval_envs": 1,
            "max_gradient_norm": 0.5,
            "seed": 0,
        }
    )

  # For rollout/loading we don't need many parallel envs; use 1 to save memory.
  # This only affects how Brax initializes the training wrapper, not the
  # already-trained policy parameters.
  ppo_params["num_envs"] = 1
  ppo_params["num_eval_envs"] = 1
  if "batch_size" in ppo_params:
    ppo_params["batch_size"] = 1
  return ppo_params


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy_from_checkpoint(
    ckpt_dir: epath.Path,
    env_name: str = "PhonebotJoystickFlatTerrain",
    env_config_path: str | None = None,
):
  """Load joystick policy from a specific checkpoint directory."""
  # Build environment config and instance.
  EnvClass, default_config = env_registry.get_environment(env_name)
  env_cfg = default_config()

  # Optional: merge in environment overrides from a YAML config file (for
  # example, a previously saved `env_config.yaml` from a training run).
  if env_config_path:
    cfg_path = epath.Path(env_config_path).expanduser()
    if not cfg_path.is_absolute():
      cfg_path = epath.Path(_THIS_DIR) / cfg_path
    cfg_path = cfg_path.resolve()
    try:
      import yaml  # local import to avoid affecting startup time

      with open(cfg_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
      if isinstance(loaded, dict):
        env_cfg.update(loaded)
      print(f"[LOAD] Applied env_config overrides from: {cfg_path}")
    except Exception as e:  # pylint: disable=broad-except
      print(f"Warning: failed to load env_config from {cfg_path}: {e}")
  else:
    print("[LOAD] No env_config_path provided; using default env_cfg from registry.")

  # For joystick environments, explicitly choose the task so that we load the
  # correct scene XML (flat vs rough). Other envs keep their own defaults.
  if env_name.endswith("FlatTerrain"):
    task = "flat_terrain"
  elif env_name.endswith("RoughTerrain"):
    task = "rough_terrain"
  else:
    raise ValueError(f"Unexpected env_name for joystick rollout: {env_name}")

  env = EnvClass(task=task, config=env_cfg)

  # PPO params for architecture etc.; no training, just restore.
  ppo_params = get_ppo_params_for_hi()
  ppo_params["num_timesteps"] = 0

  ppo_training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params["network_factory"]
    )

  train_fn = functools.partial(
      ppo.train,
      **ppo_training_params,
      network_factory=network_factory,
      progress_fn=None,
  )

  make_inference_fn, params, _ = train_fn(
      environment=env,
      eval_env=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      restore_checkpoint_path=ckpt_dir,
  )
  print(f"[LOAD] {env_name} from {ckpt_dir}")
  return env, env_cfg, make_inference_fn, params


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_joystick(
    env,
    env_cfg,
    make_inference_fn,
    params,
    out: str,
    command=(1.0, 0.0, 0.0),
    render_every: int = 1,
    anim_skip: int = 10,
    animate_out: str | None = None,
    max_steps: int = 1000,
    start_pose=(0.0, 0.0, 0.0),
    camera: str | None = "track",
    show_contact: bool = False,
):
  """Generate rollout video with joystick commands."""
  policy = jax.jit(make_inference_fn(params, deterministic=True))
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  rng = jax.random.PRNGKey(0)
  state = jit_reset(rng)

  # Optionally set initial (x, y, yaw) pose before rollout.
  if start_pose is not None:
    x0, y0, yaw0 = start_pose
    qpos = state.data.qpos
    qvel = state.data.qvel
    qpos = qpos.at[0].set(jp.asarray(x0, jp.float32))
    qpos = qpos.at[1].set(jp.asarray(y0, jp.float32))
    quat = jp.array(
        [jp.cos(yaw0 / 2.0), 0.0, 0.0, jp.sin(yaw0 / 2.0)],
        dtype=jp.float32,
    )
    qpos = qpos.at[3:7].set(quat)
    qvel = jp.zeros_like(qvel)
    new_data = state.data.replace(qpos=qpos, qvel=qvel)
    state = state.replace(data=new_data)

  cmd = jp.array(command)

  traj = []
  torso_body_id = getattr(env, "_torso_body_id", None)
  if torso_body_id is None:
    try:
      torso_body_id = int(env.mj_model.body("base_motor_link").id)
    except Exception:  # pylint: disable=broad-except
      torso_body_id = 1
  imu_site_id = getattr(env, "_site_id", None)
  anim_positions: list[list[float]] = []
  anim_yaws: list[float] = []

  print(f"Starting rollout '{out}' for up to {max_steps} steps with command {command}")
  for step in range(max_steps):
    # Set joystick command each step (vx, vy, yaw).
    state.info["command"] = cmd

    act_rng, rng = jax.random.split(rng)
    ctrl, _ = policy(state.obs, act_rng)
    state = jit_step(state, ctrl)
    traj.append(state)

    # Store pose for 2D animation every anim_skip steps.
    if (step % anim_skip) == 0:
      s = state
      # Use IMU site pose (x-forward convention) when available.
      if imu_site_id is not None:
        xyz = s.data.site_xpos[int(imu_site_id)]
        x_axis = s.data.site_xmat[int(imu_site_id), 0]
      else:
        xyz = s.data.xpos[int(torso_body_id)]
        x_axis = s.data.xmat[int(torso_body_id), 0]
      yaw = -jp.arctan2(x_axis[1], x_axis[0])
      pos = jp.array([xyz[0], xyz[1], yaw], dtype=jp.float32)
      pos_np = np.array(pos)
      anim_positions.append([float(pos_np[0]), float(pos_np[1])])
      anim_yaws.append(float(pos_np[2]))

    if bool(state.done):
      print(f"Episode terminated at step {step}")
      break

  # Render to video using Mujoco Playground's render API.
  traj_to_render = traj[::render_every]
  fps = 1.0 / env.dt / render_every
  scene_option = mujoco.MjvOption()
  if show_contact:
    # Visualize contact points and forces for debugging ground reaction forces.
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
  # If camera is None, use the default view (uses <visual><global ...>).
  # Some of our generated scenes do not define named cameras. If the requested
  # camera doesn't exist, fall back to the default camera (camera=None).
  requested_camera = camera
  if requested_camera is not None:
    try:
      # Raises KeyError if camera name doesn't exist.
      env.mj_model.camera(str(requested_camera))
    except Exception:  # pylint: disable=broad-except
      try:
        cam_names = []
        for cam_id in range(int(env.mj_model.ncam)):
          name = mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
          if name:
            cam_names.append(name)
      except Exception:  # pylint: disable=broad-except
        cam_names = []
      print(
          f"[RENDER] Warning: camera {requested_camera!r} not found in model; "
          f"falling back to default camera (None). Available cameras: {cam_names}"
      )
      requested_camera = None

  try:
    frames = env.render(
        traj_to_render,
        camera=requested_camera,
        width=640,
        height=480,
        scene_option=scene_option,
    )
  except ValueError as e:
    # Extra safety: if MuJoCo still complains about a missing named camera,
    # retry with default camera.
    msg = str(e)
    if (requested_camera is not None) and ("camera" in msg) and ("does not exist" in msg):
      print(
          f"[RENDER] Warning: render failed for camera {requested_camera!r} ({e}); "
          "retrying with default camera (None)."
      )
      frames = env.render(
          traj_to_render, camera=None, width=640, height=480, scene_option=scene_option
      )
    else:
      raise
  media.write_video(out, frames, fps=fps)
  print(f"[VIDEO] wrote {out}")

  # 2D trajectory animation (same style as T1 loader).
  if animate_out is None:
    if out.lower().endswith(".mp4"):
      animate_out = out[:-4] + "_animated.mp4"
    else:
      animate_out = out + "_animated.mp4"

  if len(anim_positions) >= 2:
    positions = np.asarray(anim_positions)
    yaws = np.asarray(anim_yaws)
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
    dx = max(1.0, (max_x - min_x) * 0.2)
    dy = max(1.0, (max_y - min_y) * 0.2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{env.__class__.__name__}: 2D Robot Trajectory")
    ax.set_xlim(min_x - dx, max_x + dx)
    ax.set_ylim(min_y - dy, max_y + dy)

    robot_point, = ax.plot([], [], "bo", markersize=6, label="Robot")
    traj_line, = ax.plot([], [], "b-", linewidth=2, alpha=0.7, label="Trajectory")
    arrow_len = 0.5
    arrow = ax.arrow(
        positions[0, 0],
        positions[0, 1],
        arrow_len,
        0.0,
        head_width=0.15,
        head_length=0.2,
        fc="blue",
        ec="blue",
        alpha=0.7,
    )
    ax.legend(loc="upper right")

    def _animate(i):
      nonlocal arrow
      i = int(i)
      x, y = positions[i, 0], positions[i, 1]
      yaw = yaws[i]
      dx_a = arrow_len * np.cos(yaw)
      dy_a = arrow_len * np.sin(yaw)
      robot_point.set_data([x], [y])
      traj_line.set_data(positions[: i + 1, 0], positions[: i + 1, 1])
      arrow.remove()
      arrow = ax.arrow(
          x,
          y,
          dx_a,
          dy_a,
          head_width=0.15,
          head_length=0.2,
          fc="blue",
          ec="blue",
          alpha=0.7,
      )
      return robot_point, traj_line, arrow

    anim = animation.FuncAnimation(
        fig,
        _animate,
        frames=len(positions),
        interval=50,
        blit=False,
        repeat=True,
    )
    anim.save(
        animate_out,
        writer="ffmpeg",
        fps=max(2, int(20 // max(1, anim_skip // 5))),
    )
    plt.close(fig)
    print(f"[ANIMATION] wrote {animate_out}")


def main():
  print("=" * 80)
  print("LOAD AND ROLLOUT PHONEBOT JOYSTICK POLICY")
  print("=" * 80)

  # Optional config path: python load_and_rollout_Hi.py [config_path]
  cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
      _THIS_DIR, "rollout_phonebot_joystick_config.yaml"
  )
  print(f"Using rollout config: {cfg_path}")

  with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  # Resolve checkpoint directory (relative paths are taken from this script dir).
  ckpt_dir = epath.Path(cfg["checkpoint_dir"])
  if not ckpt_dir.is_absolute():
    ckpt_dir = epath.Path(_THIS_DIR) / ckpt_dir
  ckpt_dir = ckpt_dir.resolve()
  print(f"Using checkpoint directory: {ckpt_dir}")

  if not ckpt_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

  env_name = cfg.get("env_name", "PhonebotJoystickFlatTerrain")
  env_cfg_path = cfg.get("env_config_path", None)
  if env_cfg_path is not None:
    print(f"Using env_config_path from config: '{env_cfg_path}'")
  env, env_cfg, make_inf, params = load_policy_from_checkpoint(
      ckpt_dir,
      env_name=env_name,
      env_config_path=env_cfg_path,
  )

  output_dir = cfg.get("output_dir", "video_rollout_phonebot")
  os.makedirs(output_dir, exist_ok=True)
  print(f"Output directory: {output_dir}")

  # Global defaults.
  default_max_steps = int(cfg.get("max_steps", 1000))
  default_render_every = int(cfg.get("render_every", 1))
  default_anim_skip = int(cfg.get("anim_skip", 10))
  default_show_contact = bool(cfg.get("show_contact", False))

  print("\nGenerating rollout videos from config...")
  rollouts = cfg.get("rollouts", [])
  for r in rollouts:
    name = r.get("name", "unnamed")
    out_name = r.get("out", f"{name}.mp4")
    out_path = os.path.join(output_dir, out_name)

    command = tuple(r.get("command", [1.0, 0.0, 0.0]))
    start_pose = tuple(r.get("start_pose", [0.0, 0.0, 0.0]))
    camera = r.get("camera", "track")
    show_contact = bool(r.get("show_contact", default_show_contact))
    max_steps = int(r.get("max_steps", default_max_steps))
    render_every = int(r.get("render_every", default_render_every))
    anim_skip = int(r.get("anim_skip", default_anim_skip))

    print(
        f"  - {name}: out={out_path}, command={command}, camera={camera}, "
        f"show_contact={show_contact}"
    )
    rollout_joystick(
        env,
        env_cfg,
        make_inf,
        params,
        out=out_path,
        command=command,
        start_pose=start_pose,
        camera=camera,
        max_steps=max_steps,
        render_every=render_every,
        anim_skip=anim_skip,
        show_contact=show_contact,
    )

  print(f"\n✅ All videos saved to {output_dir}/")


if __name__ == "__main__":
  main()


