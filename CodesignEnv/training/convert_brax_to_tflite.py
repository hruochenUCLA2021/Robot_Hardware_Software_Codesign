#!/usr/bin/env python3
"""
Convert a trained Brax PPO actor policy to a .tflite file (for Android / LiteRT).

This script:
- Loads a checkpoint from CodesignEnv/training/checkpoints/...
- Builds the same joystick environment used for training
- Restores PPO params (including observation normalizer)
- Exports the actor policy as a TF Lite model:
    input : obs["state"]  (float32)
    output: action        (float32)

Usage:
  python convert_brax_to_tflite.py
  python convert_brax_to_tflite.py convert_brax_to_tflite_config.yaml
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

# Configure JAX memory behavior early.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import functools

import jax
import jax.numpy as jp
import yaml
from etils import epath
from ml_collections import config_dict

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import train as ppo
from jax.experimental import jax2tf

import tensorflow as tf

from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

# Ensure project root (containing Robot_Hardware_Software_Codesign) is on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

from CodesignEnv import registry as env_registry  # noqa: E402


def _load_yaml(path: str) -> dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def _get_ppo_params_for_phonebot_export() -> config_dict.ConfigDict:
  """Reuse T1 PPO config; force policy/value obs keys."""
  ppo_params = locomotion_params.brax_ppo_config("T1JoystickFlatTerrain")
  if "network_factory" in ppo_params:
    ppo_params["network_factory"]["policy_obs_key"] = "state"
    ppo_params["network_factory"]["value_obs_key"] = "privileged_state"
  # For restore/export we only need a single env.
  ppo_params["num_envs"] = 1
  ppo_params["num_eval_envs"] = 1
  if "batch_size" in ppo_params:
    ppo_params["batch_size"] = 1
  ppo_params["num_timesteps"] = 0
  ppo_params["num_evals"] = 1
  return ppo_params


def _task_from_env_name(env_name: str) -> str:
  """Derive CodesignEnv task string from registry env name.

  This must match the task naming used by `CodesignEnv.configs.hi_constants.phonebot_task_to_xml`
  (and chopstickbot equivalent), otherwise the env will load the wrong scene XML.
  """
  env_name = str(env_name)
  is_phonebot = "Phonebot" in env_name
  is_chopstickbot = "Chopstickbot" in env_name
  robot = "phonebot" if is_phonebot else ("chopstickbot" if is_chopstickbot else "unknown")

  if "FlatTerrain" in env_name:
    task = "flat_terrain"
  elif "RoughTerrain" in env_name:
    task = "rough_terrain"
  else:
    raise ValueError(f"Unexpected env_name: {env_name}")

  # IMU variants.
  if "AlterFV2" in env_name:
    task = f"{task}_alternative_imu_fv2"
  elif "Alter" in env_name:
    task = f"{task}_alternative_imu"

  # Torque-aware variants.
  if "TorqueAwared" in env_name:
    if robot == "phonebot":
      # Phonebot torque-aware envs in this repo are FV2-based.
      if "AlterFV2" not in env_name:
        raise ValueError(
            "Phonebot TorqueAwared envs expect AlterFV2 task naming in this repo. "
            f"Got env_name={env_name!r}"
        )
    elif robot == "chopstickbot":
      # Chopstickbot torque-aware envs are Alter-based (no FV2 in naming).
      if ("AlterFV2" not in env_name) and ("Alter" not in env_name):
        raise ValueError(
            "Chopstickbot TorqueAwared envs expect Alter task naming in this repo. "
            f"Got env_name={env_name!r}"
        )
    task = f"{task}_torque"

  # Ankle-collision variant uses separate scene XMLs.
  if "AnkleCollision" in env_name:
    if robot != "phonebot":
      raise ValueError(
          "AnkleCollision task naming is only supported for Phonebot in this repo. "
          f"Got env_name={env_name!r}"
      )
    task = f"{task}_ankle_collision"

  return task


def _load_policy(
    ckpt_dir: epath.Path,
    *,
    env_name: str,
    env_config_path: str | None,
):
  ckpt_dir = _resolve_ckpt_leaf_dir(epath.Path(ckpt_dir))
  EnvClass, default_config = env_registry.get_environment(env_name)
  env_cfg = default_config()

  if env_config_path:
    cfg_path = epath.Path(env_config_path).expanduser()
    if not cfg_path.is_absolute():
      cfg_path = epath.Path(_THIS_DIR) / cfg_path
    cfg_path = cfg_path.resolve()
    loaded = _load_yaml(str(cfg_path))
    if isinstance(loaded, dict):
      env_cfg.update(loaded)
    # Backward/portability fix: older training runs sometimes saved absolute
    # `motor_params_path` (machine-specific). If that absolute path does not
    # exist on this machine, try falling back to a relative filename under
    # `CodesignEnv/configs/`, which the env resolves automatically.
    try:
      mpp = str(getattr(env_cfg, "motor_params_path", "") or "")
      if mpp.startswith("/") and (not epath.Path(mpp).expanduser().exists()):
        cfgs_dir = (epath.Path(_THIS_DIR).parent / "configs").resolve()
        cand = cfgs_dir / epath.Path(mpp).name
        if cand.exists():
          env_cfg["motor_params_path"] = cand.name
          print(f"[LOAD] Rewrote motor_params_path to relative: {cand.name}")
    except Exception:  # pylint: disable=broad-except
      pass
    print(f"[LOAD] Applied env_config overrides from: {cfg_path}")
  else:
    print("[LOAD] No env_config_path provided; using default env_cfg from registry.")

  env = EnvClass(task=_task_from_env_name(env_name), config=env_cfg)

  ppo_params = _get_ppo_params_for_phonebot_export()
  ppo_training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    # brax.ppo.train expects network_factory passed as kwarg, not embedded.
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params["network_factory"]
    )

  train_fn = functools.partial(
      ppo.train,
      **ppo_training_params,
      network_factory=network_factory,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      progress_fn=None,
  )

  make_inference_fn, params, _ = train_fn(
      environment=env,
      eval_env=env,
      restore_checkpoint_path=ckpt_dir,
  )
  print(f"[LOAD] {env_name} from {ckpt_dir}")
  normalize_observations = bool(ppo_params.get("normalize_observations", True))
  return env, env_cfg, make_inference_fn, params, network_factory, normalize_observations


def _is_orbax_leaf_dir(p: epath.Path) -> bool:
  """Heuristic: does this directory look like an Orbax checkpoint leaf?"""
  if not p.exists() or not p.is_dir():
    return False
  # Typical Orbax artifacts (OCDBT format).
  return (p / "_CHECKPOINT_METADATA").exists() or (p / "_METADATA").exists() or (p / "manifest.ocdbt").exists()


def _resolve_ckpt_leaf_dir(p: epath.Path) -> epath.Path:
  """Accept either leaf dir or a parent containing `final/` or numeric leaves."""
  p = epath.Path(p)
  try:
    if not p.is_absolute():
      p = p.resolve()
  except Exception:
    pass
  if _is_orbax_leaf_dir(p):
    return p
  if (p / "final").exists() and _is_orbax_leaf_dir(p / "final"):
    return p / "final"
  # Common layout: parent contains numeric checkpoint subdirs (0, 1000000, ...)
  try:
    if p.exists() and p.is_dir():
      numeric_leaves: list[tuple[int, epath.Path]] = []
      for child in p.iterdir():
        if not child.is_dir():
          continue
        name = child.name
        if name.isdigit() and _is_orbax_leaf_dir(child):
          numeric_leaves.append((int(name), child))
      if numeric_leaves:
        numeric_leaves.sort(key=lambda x: x[0])
        return numeric_leaves[-1][1]
  except Exception:
    pass
  return p


def _maybe_write_ppo_network_config_json(
    *,
    ckpt_dir: epath.Path,
    mode: str,
    observation_size: int,
    action_size: int,
    normalize_observations: bool,
    network_factory,
):
  """Write/replace `ppo_network_config.json` next to an Orbax checkpoint leaf."""
  mode = str(mode or "skip").lower()
  if mode not in ("skip", "if_missing", "replace"):
    raise ValueError("export.ppo_network_config_json.mode must be one of: skip, if_missing, replace")
  if mode == "skip":
    return

  ckpt_dir = _resolve_ckpt_leaf_dir(epath.Path(ckpt_dir))
  out_path = ckpt_dir / "ppo_network_config.json"
  if mode == "if_missing" and out_path.exists():
    print(f"[JSON] exists, leaving as-is: {out_path}")
    return

  cfg_json = ppo_checkpoint.network_config(
      observation_size=int(observation_size),
      action_size=int(action_size),
      normalize_observations=bool(normalize_observations),
      network_factory=network_factory,
  )
  out_path.write_text(cfg_json.to_json_best_effort(), encoding="utf-8")
  print(f"[JSON] wrote: {out_path}")


def _infer_expected_state_dim(policy_raw, state_obs: dict[str, jp.ndarray]) -> int:
  """Return obs['state'] dim expected by checkpoint (handles old checkpoints)."""
  expected_dim = int(state_obs["state"].shape[-1])
  try:
    _ = policy_raw(state_obs, jax.random.PRNGKey(0))
    return expected_dim
  except Exception as e:  # pylint: disable=broad-except
    msg = str(e)
    m = re.search(r"broadcasting: \\((\\d+),\\), \\((\\d+),\\)", msg)
    if not m:
      raise
    return int(m.group(2))


def _adapt_state_vec(x: jp.ndarray, target_dim: int) -> jp.ndarray:
  cur = int(x.shape[-1])
  if cur == int(target_dim):
    return x
  if cur > int(target_dim):
    return x[..., : int(target_dim)]
  pad = int(target_dim) - cur
  if x.ndim == 1:
    return jp.pad(x, ((0, pad),))
  if x.ndim == 2:
    return jp.pad(x, ((0, 0), (0, pad)))
  raise ValueError(f"Expected state_vec rank 1 or 2, got shape {x.shape}")


def export_actor_tflite(
    *,
    ckpt_dir: epath.Path,
    env_name: str,
    env_config_path: str | None,
    out_dir: epath.Path,
    out_tflite: str,
    out_metadata_json: str,
    dry_run: bool,
    input_mode: str,
    use_select_tf_ops: bool,
    optimize: bool,
    ppo_network_config_json_mode: str,
):
  env, env_cfg, make_inf, params, network_factory, normalize_observations = _load_policy(
      ckpt_dir, env_name=env_name, env_config_path=env_config_path
  )
  policy_raw = make_inf(params, deterministic=True)

  # Determine input size expected by this checkpoint.
  reset_state = jax.jit(env.reset)(jax.random.PRNGKey(0))
  expected_state_dim = _infer_expected_state_dim(policy_raw, reset_state.obs)
  nu = int(env.mjx_model.nu)

  # Also expose default pose (used when converting action -> target joints).
  default_pose = None
  if hasattr(env, "_default_pose"):
    default_pose = [float(x) for x in list(env._default_pose)]

  print(f"[EXPORT] env_name={env_name}")
  print(f"[EXPORT] expected obs['state'] dim = {expected_state_dim}")
  print(f"[EXPORT] action dim (nu) = {nu}")

  _maybe_write_ppo_network_config_json(
      ckpt_dir=ckpt_dir,
      mode=ppo_network_config_json_mode,
      observation_size=expected_state_dim,
      action_size=nu,
      normalize_observations=normalize_observations,
      network_factory=network_factory,
  )

  if dry_run:
    return

  # Build a pure JAX function: state_vec -> action_vec.
  rng0 = jax.random.PRNGKey(0)

  def jax_actor(state_vec: jp.ndarray) -> jp.ndarray:
    state_vec = state_vec.astype(jp.float32)
    state_vec = _adapt_state_vec(state_vec, expected_state_dim)
    obs = {"state": state_vec}
    act, _ = policy_raw(obs, rng0)
    return act.astype(jp.float32)

  # Convert to TF.
  tf_actor = jax2tf.convert(jax_actor, with_gradient=False)

  if input_mode == "batch":
    input_spec = tf.TensorSpec(shape=(1, expected_state_dim), dtype=tf.float32, name="state")
  elif input_mode == "vector":
    input_spec = tf.TensorSpec(shape=(expected_state_dim,), dtype=tf.float32, name="state")
  else:
    raise ValueError("export.input_mode must be 'vector' or 'batch'")

  @tf.function(input_signature=[input_spec])
  def tf_module(state):  # pylint: disable=unused-argument
    return tf_actor(state)

  concrete_fn = tf_module.get_concrete_function()
  converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

  if use_select_tf_ops:
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
  else:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

  if optimize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

  out_dir = epath.Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  tflite_path = out_dir / out_tflite
  metadata_path = out_dir / out_metadata_json

  tflite_model = converter.convert()
  tflite_path.write_bytes(tflite_model)
  print(f"[EXPORT] wrote {tflite_path}")

  metadata = {
      "env_name": env_name,
      "checkpoint_dir": str(ckpt_dir),
      "policy_obs_key": "state",
      "value_obs_key": "privileged_state",
      "state_dim": expected_state_dim,
      "action_dim": nu,
      "default_pose": default_pose,
      "note": "Action is normalized; map to joint targets as default_pose + action * action_scale.",
  }
  metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(f"[EXPORT] wrote {metadata_path}")


def main():
  cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
      _THIS_DIR, "convert_brax_to_tflite_config.yaml"
  )
  cfg = _load_yaml(cfg_path)
  job_name = cfg.get("job_name", "")
  jobs = cfg.get("jobs", {}) or {}
  if not jobs:
    raise ValueError("No jobs found in config under `jobs:`")

  # Allow job_name as:
  # - string: one job
  # - list: multiple jobs
  # - "all": run all jobs
  job_names: list[str]
  if isinstance(job_name, list):
    job_names = [str(x).strip() for x in job_name if str(x).strip()]
  else:
    job_name_str = str(job_name).strip()
    if job_name_str.lower() == "all":
      job_names = sorted(list(jobs.keys()))
    else:
      job_names = [job_name_str] if job_name_str else []

  if not job_names:
    raise ValueError(f"job_name must be a string, a list of job keys, or 'all'. Available jobs: {sorted(list(jobs.keys()))}")

  missing = [n for n in job_names if n not in jobs]
  if missing:
    raise ValueError(f"Unknown job_name(s): {missing}. Available jobs: {sorted(list(jobs.keys()))}")

  export_cfg = cfg.get("export", {}) or {}
  dry_run = bool(export_cfg.get("dry_run", False))
  input_mode = str(export_cfg.get("input_mode", "vector"))
  use_select_tf_ops = bool(export_cfg.get("use_select_tf_ops", True))
  optimize = bool(export_cfg.get("optimize", False))
  ppo_network_config_json_cfg = export_cfg.get("ppo_network_config_json", {}) or {}
  ppo_network_config_json_mode = str(ppo_network_config_json_cfg.get("mode", "skip"))
  continue_on_error = bool(export_cfg.get("continue_on_error", True))

  print(f"Using config: {cfg_path}")
  print(f"Jobs: {job_names}")

  failures: list[tuple[str, str]] = []
  for name in job_names:
    job = jobs[name]
    ckpt_dir = epath.Path(job["checkpoint_dir"])
    if not ckpt_dir.is_absolute():
      ckpt_dir = epath.Path(_THIS_DIR) / ckpt_dir
    ckpt_dir = ckpt_dir.resolve()
    if not ckpt_dir.exists():
      msg = f"Checkpoint directory not found: {ckpt_dir}"
      if continue_on_error:
        print(f"[SKIP] {name}: {msg}")
        failures.append((name, msg))
        continue
      raise FileNotFoundError(msg)

    env_name = str(job.get("env_name", "PhonebotJoystickFlatTerrain"))
    env_config_path = job.get("env_config_path", None)
    out_dir = job.get("out_dir", "exported_tflite")
    out_tflite = str(job.get("out_tflite", f"{name}.tflite"))
    out_metadata_json = str(job.get("out_metadata_json", f"{name}_metadata.json"))

    print("\n" + "=" * 80)
    print(f"[JOB] {name}")
    print("=" * 80)
    try:
      export_actor_tflite(
          ckpt_dir=ckpt_dir,
          env_name=env_name,
          env_config_path=env_config_path,
          out_dir=epath.Path(_THIS_DIR) / out_dir,
          out_tflite=out_tflite,
          out_metadata_json=out_metadata_json,
          dry_run=dry_run,
          input_mode=input_mode,
          use_select_tf_ops=use_select_tf_ops,
          optimize=optimize,
          ppo_network_config_json_mode=ppo_network_config_json_mode,
      )
    except Exception as e:  # pylint: disable=broad-except
      msg = f"{type(e).__name__}: {e}"
      if continue_on_error:
        print(f"[FAIL] {name}: {msg}")
        failures.append((name, msg))
        continue
      raise

  if failures:
    print("\n" + "=" * 80)
    print("[DONE] Completed with some failures:")
    for n, m in failures:
      print(f"- {n}: {m}")
    print("=" * 80)


if __name__ == "__main__":
  main()

