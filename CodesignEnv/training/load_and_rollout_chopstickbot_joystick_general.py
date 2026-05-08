#!/usr/bin/env python3
"""
General Chopstickbot rollout (MuJoCo CPU) recording videos.

- Explicit `xml_path` + `policy_path` in YAML.
- MuJoCo CPU simulation at (physics_hz/control_hz), record MP4 via `mediapy`.
- Policy loading:
  - Brax: try `ppo_checkpoint.load_policy` first; fallback to `ppo.train(... restore_checkpoint_path=...)`
  - TFLite: LiteRT (`ai-edge-litert`) or TensorFlow backend.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
from typing import Any, Callable

import numpy as np
import yaml
import mediapy as media
import mujoco
from etils import epath

# Configure before importing JAX/TF.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# training/ -> CodesignEnv/ -> Robot_Hardware_Software_Codesign/
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

from CodesignEnv import registry as env_registry  # noqa: E402


def _load_yaml(path: str) -> dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def _as_abs_path(p: str, *, base_dir: str) -> epath.Path:
  x = epath.Path(p).expanduser()
  if x.is_absolute():
    return x
  # Primary: resolve relative to base_dir.
  p1 = (epath.Path(base_dir) / x).resolve()
  if p1.exists():
    return p1
  # Fallback: some configs include "Robot_Hardware_Software_Codesign/..." even
  # though base_dir already points at that folder.
  workspace_root = epath.Path(base_dir).parent
  p2 = (workspace_root / x).resolve()
  # Only use fallback if it actually exists. Otherwise keep the primary path
  # (important for output directories that don't exist yet).
  return p2 if p2.exists() else p1


def _is_orbax_leaf_dir(p: epath.Path) -> bool:
  if not p.exists() or not p.is_dir():
    return False
  return (
      (p / "_CHECKPOINT_METADATA").exists()
      or (p / "_METADATA").exists()
      or (p / "manifest.ocdbt").exists()
  )


def _resolve_ckpt_leaf_dir(p: epath.Path) -> epath.Path:
  p = epath.Path(p).expanduser()
  if not p.is_absolute():
    p = (epath.Path(_THIS_DIR) / p).resolve()
  if _is_orbax_leaf_dir(p):
    return p
  if (p / "final").exists() and _is_orbax_leaf_dir(p / "final"):
    return p / "final"
  numeric = []
  for c in p.iterdir():
    if c.is_dir() and c.name.isdigit() and _is_orbax_leaf_dir(c):
      numeric.append((int(c.name), c))
  if numeric:
    numeric.sort(key=lambda x: x[0])
    return numeric[-1][1]
  return p


def _build_mjx_env(
    env_name: str | None,
    task: str | None,
    env_config_path: str | None,
    env_config_overrides: dict | None,
):
  if not env_name:
    return None
  if not task:
    raise ValueError("When env_name is provided, task must also be provided.")
  EnvClass, default_config = env_registry.get_environment(env_name)
  env_cfg = default_config()
  if env_config_path:
    cfg_path = _as_abs_path(str(env_config_path), base_dir=_THIS_DIR)
    loaded = _load_yaml(str(cfg_path))
    if isinstance(loaded, dict):
      env_cfg.update(loaded)
    print(f"[ENV] Applied env_config from: {cfg_path}")
  if env_config_overrides:
    if isinstance(env_config_overrides, dict):
      env_cfg.update(env_config_overrides)
    print("[ENV] Applied env_config_overrides (dict).")
  return EnvClass(task=task, config=env_cfg)


def _get_restore_ppo_params() -> dict[str, Any]:
  ppo_params = locomotion_params.brax_ppo_config("T1JoystickFlatTerrain")
  if "network_factory" in ppo_params:
    ppo_params["network_factory"]["policy_obs_key"] = "state"
    ppo_params["network_factory"]["value_obs_key"] = "privileged_state"
  ppo_params["num_envs"] = 1
  ppo_params["num_eval_envs"] = 1
  if "batch_size" in ppo_params:
    ppo_params["batch_size"] = 1
  ppo_params["num_timesteps"] = 0
  ppo_params["num_evals"] = 1
  return dict(ppo_params)


def _wrap_brax_policy_with_dim_adapt(policy_raw) -> Callable[[np.ndarray], np.ndarray]:
  expected_dim: dict[str, int] = {"n": -1}

  def _adapt(x: np.ndarray) -> np.ndarray:
    n = int(expected_dim["n"])
    if n <= 0:
      return x
    if int(x.shape[-1]) == n:
      return x
    if int(x.shape[-1]) > n:
      return x[:n]
    pad = n - int(x.shape[-1])
    return np.pad(x, (0, pad), mode="constant")

  def act_fn(state_vec: np.ndarray) -> np.ndarray:
    x = np.asarray(state_vec, dtype=np.float32).reshape((-1,))
    x = _adapt(x)
    obs = {"state": jp.asarray(x, dtype=jp.float32)}
    try:
      a, _ = policy_raw(obs, jax.random.PRNGKey(0))
    except Exception as e:  # pylint: disable=broad-except
      msg = str(e)
      m = re.search(r"broadcasting.*\\((\\d+),\\).*\\((\\d+),\\)", msg)
      if not m:
        raise
      expected_dim["n"] = int(m.group(2))
      x2 = _adapt(np.asarray(state_vec, dtype=np.float32).reshape((-1,)))
      obs2 = {"state": jp.asarray(x2, dtype=jp.float32)}
      a, _ = policy_raw(obs2, jax.random.PRNGKey(0))
    return np.asarray(a, dtype=np.float32).reshape((-1,))

  return act_fn


def _load_brax_policy(
    policy_path: str,
    *,
    env_name: str | None,
    task: str | None,
    env_config_path: str | None,
    env_config_overrides: dict | None,
) -> tuple[Callable[[np.ndarray], np.ndarray], str, epath.Path, Any]:
  ckpt_leaf = _resolve_ckpt_leaf_dir(epath.Path(policy_path))
  if not ckpt_leaf.exists():
    raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_leaf}")

  json_path = ckpt_leaf / "ppo_network_config.json"
  if json_path.exists():
    try:
      print(f"[POLICY][BRAX] Trying ppo_checkpoint.load_policy ({json_path})")
      pol = ppo_checkpoint.load_policy(ckpt_leaf, deterministic=True)
      pol = jax.jit(pol)
      return _wrap_brax_policy_with_dim_adapt(pol), "ppo_checkpoint.load_policy", ckpt_leaf, None
    except Exception as e:  # pylint: disable=broad-except
      print(f"[POLICY][BRAX] load_policy failed; fallback to restore. {type(e).__name__}: {e}")

  env = _build_mjx_env(env_name, task, env_config_path, env_config_overrides)
  if env is None:
    raise ValueError("Brax fallback restore requires env_name+task.")

  ppo_params = _get_restore_ppo_params()
  train_kwargs = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del train_kwargs["network_factory"]
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params["network_factory"])

  train_fn = functools.partial(
      ppo.train,
      **train_kwargs,
      network_factory=network_factory,
      progress_fn=None,
  )
  make_inf, params, _ = train_fn(
      environment=env,
      eval_env=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      restore_checkpoint_path=ckpt_leaf,
  )
  pol = jax.jit(make_inf(params, deterministic=True))
  return _wrap_brax_policy_with_dim_adapt(pol), "ppo.train(restore_checkpoint_path=...)", ckpt_leaf, env


def _load_tflite_policy(
    policy_path: str,
    *,
    backend: str,
    num_threads: int,
) -> tuple[Callable[[np.ndarray], np.ndarray], str, epath.Path]:
  p = _as_abs_path(policy_path, base_dir=_THIS_DIR)
  if not p.exists():
    raise FileNotFoundError(f"TFLite model not found: {p}")

  backend = str(backend or "litert").lower().strip()
  if backend == "litert":
    from ai_edge_litert.interpreter import Interpreter  # type: ignore

    interp = Interpreter(model_path=str(p), num_threads=int(num_threads))
  elif backend == "tensorflow":
    import tensorflow as tf  # pylint: disable=import-error

    interp = tf.lite.Interpreter(model_path=str(p), num_threads=int(num_threads))
  else:
    raise ValueError("tflite_backend must be litert|tensorflow")

  interp.allocate_tensors()
  in_details = interp.get_input_details()
  out_details = interp.get_output_details()
  if len(in_details) != 1 or len(out_details) != 1:
    raise ValueError("Expected 1 input + 1 output tensor for TFLite policy.")
  idx_in = int(in_details[0]["index"])
  idx_out = int(out_details[0]["index"])
  in_shape = tuple(int(s) for s in in_details[0]["shape"])

  def act_fn(state_vec: np.ndarray) -> np.ndarray:
    x = np.asarray(state_vec, dtype=np.float32).reshape((-1,))
    if len(in_shape) == 2:
      x = x.reshape((1, -1))
    if tuple(x.shape) != tuple(in_shape):
      raise ValueError(f"TFLite input shape mismatch: expects {in_shape}, got {x.shape}")
    interp.set_tensor(idx_in, x)
    interp.invoke()
    y = interp.get_tensor(idx_out)
    return np.asarray(y, dtype=np.float32).reshape((-1,))

  return act_fn, f"tflite({backend})", p


def _soft_joint_limits(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  lowers = m.jnt_range[1:, 0]
  uppers = m.jnt_range[1:, 1]
  c = (lowers + uppers) / 2.0
  r = uppers - lowers
  return (c - 0.5 * r * 0.95).astype(np.float32), (c + 0.5 * r * 0.95).astype(np.float32)


def _safe_sensor(d: mujoco.MjData, name: str, size: int) -> np.ndarray:
  try:
    x = np.array(d.sensor(name).data, dtype=np.float32).reshape((-1,))
    if x.size == size:
      return x
  except Exception:
    pass
  return np.zeros((size,), dtype=np.float32)


def _safe_site_xmat(m: mujoco.MjModel, d: mujoco.MjData, site_name: str) -> np.ndarray:
  try:
    sid = int(m.site(site_name).id)
    return np.array(d.site_xmat[sid], dtype=np.float32).reshape((3, 3))
  except Exception:
    return np.eye(3, dtype=np.float32)


def _compute_state_obs(
    *,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    default_pose: np.ndarray,
    last_act: np.ndarray,
    phase: np.ndarray,
    cmd: np.ndarray,
) -> np.ndarray:
  gyro = _safe_sensor(d, "gyro", 3)
  xmat = _safe_site_xmat(m, d, "imu")
  gravity = xmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
  q = np.array(d.qpos[7:], dtype=np.float32)
  qd = np.array(d.qvel[6:], dtype=np.float32)
  phase_feat = np.concatenate([np.cos(phase), np.sin(phase)], axis=0).astype(np.float32)
  return np.hstack(
      [
          gyro,
          gravity,
          cmd.astype(np.float32),
          q - default_pose,
          qd,
          last_act.astype(np.float32),
          phase_feat,
      ]
  ).astype(np.float32)


def _render_frame(
    renderer: mujoco.Renderer,
    d: mujoco.MjData,
    *,
    camera: str | None,
    scene_option: mujoco.MjvOption | None,
) -> np.ndarray:
  try:
    renderer.update_scene(d, camera=camera, scene_option=scene_option)
  except Exception:
    renderer.update_scene(d, camera=None, scene_option=scene_option)
  return renderer.render()


def rollout_mujoco_cpu(
    *,
    xml_path: epath.Path,
    env_for_torque_controller,
    home_keyframe_name: str,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    policy_kind: str,
    out_path: epath.Path,
    command: list[float],
    start_pose: list[float],
    record_json_path: epath.Path | None,
    env_registry_name: str | None,
    max_steps: int,
    render_every: int,
    fps: int,
    camera: str | None,
    physics_hz: int,
    control_hz: int,
    action_scale: float,
    show_contact: bool,
) -> None:
  m = mujoco.MjModel.from_xml_path(str(xml_path))
  m.opt.timestep = 1.0 / float(physics_hz)
  d = mujoco.MjData(m)

  try:
    qpos0 = np.array(m.keyframe(home_keyframe_name).qpos, dtype=np.float64)
  except KeyError:
    qpos0 = np.array(m.qpos0, dtype=np.float64)

  d.qpos[:] = qpos0
  d.qvel[:] = 0.0
  if int(m.nu) > 0:
    d.ctrl[:] = 0.0

  # Start pose override (x, y, yaw) for the floating base.
  x0, y0, yaw0 = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
  d.qpos[0] = x0
  d.qpos[1] = y0
  d.qpos[3] = float(np.cos(0.5 * yaw0))  # w
  d.qpos[4] = 0.0
  d.qpos[5] = 0.0
  d.qpos[6] = float(np.sin(0.5 * yaw0))  # z

  mujoco.mj_forward(m, d)

  default_pose = np.array(d.qpos[7:], dtype=np.float32)
  soft_lowers, soft_uppers = _soft_joint_limits(m)

  # -------------------------------------------------------------------------
  # Recording (qpos/qvel/qacc/actuator_force) to JSON, compatible with
  # `training/data_plotter.py`.
  # -------------------------------------------------------------------------
  qpos_record: list[list[float]] = []
  qvel_record: list[list[float]] = []
  qacc_record: list[list[float]] = []
  actuator_force_record: list[list[float]] = []

  qpos_labels: list[str] = ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]
  qvel_labels: list[str] = ["base_vx", "base_vy", "base_vz", "base_wx", "base_wy", "base_wz"]
  for j in range(int(m.njnt)):
    if int(m.jnt_type[j]) == 0:  # mjJNT_FREE
      continue
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
    qpos_labels.append(jname)
    qvel_labels.append(jname)

  actuator_force_labels: list[str] = []
  for a in range(int(m.nu)):
    aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}"
    try:
      jid = int(m.actuator_trnid[a, 0])
      jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
      actuator_force_labels.append(f"{aname}({jname})")
    except Exception:
      actuator_force_labels.append(aname)

  motor_controller = None
  if env_for_torque_controller is not None and hasattr(env_for_torque_controller, "_motor_controller"):
    motor_controller = getattr(env_for_torque_controller, "_motor_controller")

  n_substeps = int(round(float(physics_hz) / float(control_hz)))
  n_substeps = max(1, n_substeps)

  last_act = np.zeros((int(m.nu),), dtype=np.float32)
  phase = np.array([0.0, np.pi], dtype=np.float32)
  phase_dt = float(2.0 * np.pi / float(max(1, control_hz))) * 1.5
  cmd = np.asarray(command, dtype=np.float32).reshape((3,))

  renderer = mujoco.Renderer(m, width=640, height=480)
  scene_option = None
  if show_contact:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

  frames: list[np.ndarray] = []
  for t in range(int(max_steps)):
    if float(np.linalg.norm(cmd)) > 0.01:
      phase_tp1 = phase + phase_dt
      phase = (np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi).astype(np.float32)
    else:
      phase = np.ones((2,), dtype=np.float32) * np.pi

    obs_state = _compute_state_obs(
        m=m,
        d=d,
        default_pose=default_pose,
        last_act=last_act,
        phase=phase,
        cmd=cmd,
    )
    action = np.asarray(policy_fn(obs_state), dtype=np.float32).reshape((-1,))
    last_act = action

    motor_targets = default_pose + action * float(action_scale)
    motor_targets = np.clip(motor_targets, soft_lowers, soft_uppers).astype(np.float32)

    if motor_controller is not None:
      q = jp.asarray(np.array(d.qpos[7:], dtype=np.float32))
      qd = jp.asarray(np.array(d.qvel[6:], dtype=np.float32))
      qdd = jp.asarray(np.array(d.qacc[6:], dtype=np.float32))
      tau = motor_controller.step(q, qd, qdd, jp.asarray(motor_targets))
      d.ctrl[:] = np.asarray(tau, dtype=np.float32)
    else:
      d.ctrl[:] = motor_targets.astype(np.float32)

    for _ in range(int(n_substeps)):
      mujoco.mj_step(m, d)

    if record_json_path is not None:
      qpos_record.append(np.array(d.qpos, dtype=np.float64).tolist())
      qvel_record.append(np.array(d.qvel, dtype=np.float64).tolist())
      try:
        qacc_record.append(np.array(d.qacc, dtype=np.float64).tolist())
      except Exception:
        qacc_record.append([0.0] * int(m.nv))
      try:
        actuator_force_record.append(np.array(d.actuator_force, dtype=np.float64).tolist())
      except Exception:
        actuator_force_record.append([0.0] * int(m.nu))

    if (t % int(render_every)) == 0:
      frames.append(_render_frame(renderer, d, camera=camera, scene_option=scene_option))

  out_path.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(out_path), frames, fps=int(fps))
  print(f"[VIDEO] wrote {out_path} (policy={policy_kind})")

  if record_json_path is not None:
    qacc_labels = [
        lb.replace("_v", "_a").replace("_w", "_alpha") if lb.startswith("base_") else f"{lb}_acc"
        for lb in qvel_labels
    ]
    record_data = {
        "rollout_name": str(record_json_path.stem),
        "env_registry_name": env_registry_name,
        "xml_path": str(xml_path),
        "home_keyframe_name": str(home_keyframe_name),
        "command": list(command),
        "dt": float(1.0 / float(control_hz)),
        "num_steps": int(len(qpos_record)),
        "qpos_labels": qpos_labels,
        "qvel_labels": qvel_labels,
        "qacc_labels": qacc_labels,
        "actuator_force_labels": actuator_force_labels,
        "qpos": qpos_record,
        "qvel": qvel_record,
        "qacc": qacc_record,
        "actuator_force": actuator_force_record,
    }
    record_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_json_path, "w", encoding="utf-8") as jf:
      json.dump(record_data, jf)
    print(f"[RECORD] wrote {record_json_path}")


def main() -> None:
  cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
      _THIS_DIR, "rollout_chopstickbot_joystick_general_config.yaml"
  )
  cfg = _load_yaml(cfg_path)
  print(f"Using rollout config: {cfg_path}")

  default_show_contact = bool(cfg.get("show_contact", False))
  default_record_json = bool(cfg.get("record_json", True))
  default_auto_plot = bool(cfg.get("auto_plot", True))
  default_plot_cfg = cfg.get("plot", {}) or {}

  jobs = cfg.get("jobs", {}) or {}
  if not jobs:
    raise ValueError("No jobs found in config under `jobs:`")

  job_name = cfg.get("job_name", "")
  if isinstance(job_name, list):
    job_names = [str(x).strip() for x in job_name if str(x).strip()]
  else:
    s = str(job_name).strip()
    job_names = sorted(list(jobs.keys())) if s.lower() == "all" else ([s] if s else [])
  if not job_names:
    raise ValueError(f"job_name must be a string, list, or 'all'. Available: {sorted(list(jobs.keys()))}")

  for name in job_names:
    job = jobs[name] or {}
    print("\n" + "=" * 80)
    print(f"[JOB] {name}")
    print("=" * 80)

    policy_path = str(job.get("policy_path", "")).strip()
    if not policy_path:
      raise ValueError(f"Job {name} missing policy_path")
    policy_format = str(job.get("policy_format", "auto")).lower().strip()
    if policy_format == "auto":
      policy_format = "tflite" if policy_path.lower().endswith(".tflite") else "brax"

    xml_p = _as_abs_path(str(job.get("xml_path", "")), base_dir=_PROJECT_ROOT)
    if not xml_p.exists():
      raise FileNotFoundError(f"xml_path not found: {xml_p}")

    env_name = job.get("env_name", None)
    task = job.get("task", None)
    env_config_path = job.get("env_config_path", None)
    env_config_overrides = job.get("env_config_overrides", None)
    home_keyframe_name = str(job.get("home_keyframe_name", "home"))

    output_dir = _as_abs_path(str(job.get("output_dir", "video_rollout_chopstickbot_general")), base_dir=_THIS_DIR)

    max_steps_default = int(job.get("max_steps", 1000))
    render_every_default = int(job.get("render_every", 1))
    fps = int(job.get("fps", 50))
    physics_hz = int(job.get("physics_hz", 500))
    control_hz = int(job.get("control_hz", 50))
    show_contact_default = bool(job.get("show_contact", default_show_contact))
    record_json_enable = bool(job.get("record_json", default_record_json))
    auto_plot_enable = bool(job.get("auto_plot", default_auto_plot))
    plot_cfg = job.get("plot", default_plot_cfg) or {}

    env_for_controller = None
    if env_name and task:
      # If user didn't provide env_config_overrides, default to using the same
      # XML path for the env build (useful for torque-controller reuse).
      if env_config_overrides is None:
        env_config_overrides = {"xml_path_override": str(xml_p)}
      env_for_controller = _build_mjx_env(env_name, task, env_config_path, env_config_overrides)

    action_scale = job.get("action_scale", None)
    if action_scale is None and env_for_controller is not None and hasattr(env_for_controller, "_config"):
      try:
        action_scale = float(getattr(env_for_controller._config, "action_scale"))  # pylint: disable=protected-access
        print(f"[CONFIG] action_scale from env: {action_scale}")
      except Exception:
        action_scale = 1.0
    action_scale = float(action_scale if action_scale is not None else 1.0)

    if policy_format == "tflite":
      policy_fn, policy_kind, resolved = _load_tflite_policy(
          policy_path,
          backend=str(job.get("tflite_backend", "litert")),
          num_threads=int(job.get("tflite_num_threads", 1)),
      )
      print(f"[POLICY] {policy_kind}: {resolved}")
    else:
      policy_fn, policy_kind, ckpt_leaf, _env_restore = _load_brax_policy(
          policy_path,
          env_name=env_name,
          task=task,
          env_config_path=env_config_path,
          env_config_overrides=env_config_overrides,
      )
      print(f"[POLICY] {policy_kind}: {ckpt_leaf}")

    cam_default = job.get("camera", "track")
    if cam_default is not None:
      cam_default = str(cam_default)
    try:
      _m = mujoco.MjModel.from_xml_path(str(xml_p))
      if cam_default is not None:
        _ = _m.camera(cam_default)
    except Exception:
      if cam_default is not None:
        print(f"[RENDER] Warning: camera {cam_default!r} not found; will fall back to default camera.")
        cam_default = None

    rollouts = job.get("rollouts", []) or []
    if not rollouts:
      raise ValueError(f"Job {name} has no rollouts list.")

    start_pose_default = list(job.get("start_pose_default", [0.0, 0.0, 0.0]))

    print(f"[MODEL] xml_path: {xml_p}")
    print(f"[ROLLOUT] output_dir: {output_dir}")

    recorded_jsons: list[str] = []
    for r in rollouts:
      r_name = str(r.get("name", "rollout"))
      out = str(r.get("out", f"{r_name}.mp4"))
      out_path = (output_dir / out).resolve()
      command = list(r.get("command", [1.0, 0.0, 0.0]))
      start_pose = list(r.get("start_pose", start_pose_default))
      max_steps = int(r.get("max_steps", max_steps_default))
      render_every = int(r.get("render_every", render_every_default))
      camera = r.get("camera", cam_default)
      show_contact = bool(r.get("show_contact", show_contact_default))

      # record_json can be:
      # - false: disable
      # - true/null: use default path in output_dir
      # - string: explicit path (relative -> output_dir)
      record_opt = r.get("record_json", record_json_enable)
      record_path: epath.Path | None
      if isinstance(record_opt, str) and record_opt.strip():
        rp = epath.Path(record_opt.strip())
        record_path = (output_dir / rp).resolve() if not rp.is_absolute() else rp
      elif bool(record_opt):
        record_path = (output_dir / f"{r_name}_record.json").resolve()
      else:
        record_path = None

      rollout_mujoco_cpu(
          xml_path=xml_p,
          env_for_torque_controller=env_for_controller,
          home_keyframe_name=home_keyframe_name,
          policy_fn=policy_fn,
          policy_kind=policy_kind,
          out_path=out_path,
          command=command,
          start_pose=start_pose,
          record_json_path=record_path,
          env_registry_name=str(env_name) if env_name is not None else None,
          max_steps=max_steps,
          render_every=render_every,
          fps=fps,
          camera=camera,
          physics_hz=physics_hz,
          control_hz=control_hz,
          action_scale=action_scale,
          show_contact=show_contact,
      )

      if record_path is not None:
        recorded_jsons.append(str(record_path))

    if auto_plot_enable and recorded_jsons:
      print("\n[AUTO-PLOT] Generating qpos/qvel/qacc/torque plots ...")
      from data_plotter import process_record  # noqa: E402

      for jp_path in recorded_jsons:
        if os.path.isfile(jp_path):
          process_record(jp_path, plot_cfg)
      print("[AUTO-PLOT] Done.")


if __name__ == "__main__":
  main()
