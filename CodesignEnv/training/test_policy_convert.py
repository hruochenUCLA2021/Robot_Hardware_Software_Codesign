#!/usr/bin/env python3
"""
Compare Brax (checkpoint) actor vs exported TFLite actor.

For each job:
- Load Brax PPO actor from checkpoint.
  - If `ppo_network_config.json` exists: try `ppo_checkpoint.load_policy` (fast).
  - If that fails (or JSON missing): fallback to `ppo.train(... restore_checkpoint_path=...)`.
- Load TFLite model (.tflite) via TF Lite Interpreter.
- Sample N observations (default: env.reset() with different RNG seeds).
- Feed the same obs['state'] into both policies; report diff stats.
"""

from __future__ import annotations

import os
import sys
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import functools
import json
import time

import jax
import jax.numpy as jp
import numpy as np
import tensorflow as tf
import yaml
from etils import epath
from ml_collections import config_dict

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

from CodesignEnv import registry as env_registry  # noqa: E402


def _load_yaml(path: str) -> dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def _is_orbax_leaf_dir(p: epath.Path) -> bool:
  if not p.exists() or not p.is_dir():
    return False
  return (p / "_CHECKPOINT_METADATA").exists() or (p / "_METADATA").exists() or (p / "manifest.ocdbt").exists()


def _resolve_ckpt_leaf_dir(p: epath.Path) -> epath.Path:
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
  try:
    if p.exists() and p.is_dir():
      numeric: list[tuple[int, epath.Path]] = []
      for child in p.iterdir():
        if child.is_dir() and child.name.isdigit() and _is_orbax_leaf_dir(child):
          numeric.append((int(child.name), child))
      if numeric:
        numeric.sort(key=lambda x: x[0])
        return numeric[-1][1]
  except Exception:
    pass
  return p


def _task_from_env_name(env_name: str) -> str:
  if "FlatTerrain" in env_name:
    task = "flat_terrain"
  elif "RoughTerrain" in env_name:
    task = "rough_terrain"
  else:
    raise ValueError(f"Unexpected env_name (need FlatTerrain or RoughTerrain): {env_name}")

  is_torque_awared = "TorqueAwared" in env_name
  is_alt_fv2 = "AlterFV2" in env_name
  is_alt = ("Alter" in env_name) and (not is_alt_fv2)

  if is_torque_awared:
    if is_alt_fv2:
      return f"{task}_alternative_imu_fv2_torque"
    if is_alt:
      return f"{task}_alternative_imu_torque"
    return task

  if is_alt_fv2:
    return f"{task}_alternative_imu_fv2"
  if is_alt:
    return f"{task}_alternative_imu"
  return task


def _get_restore_like_ppo_params() -> config_dict.ConfigDict:
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
  return ppo_params


def _build_env(env_name: str, env_config_path: str | None):
  EnvClass, default_config = env_registry.get_environment(env_name)
  env_cfg = default_config()
  if env_config_path:
    cfg_path = epath.Path(env_config_path).expanduser()
    if not cfg_path.is_absolute():
      cfg_path = (epath.Path(_THIS_DIR) / cfg_path).resolve()
    loaded = _load_yaml(str(cfg_path))
    if isinstance(loaded, dict):
      env_cfg.update(loaded)
    print(f"[ENV] Applied env_config from: {cfg_path}")
  env = EnvClass(task=_task_from_env_name(env_name), config=env_cfg)
  return env, env_cfg


def _load_brax_policy(
    *,
    ckpt_dir: epath.Path,
    env_name: str,
    env_config_path: str | None,
    env=None,
    env_cfg=None,
    deterministic: bool,
):
  ckpt_leaf = _resolve_ckpt_leaf_dir(ckpt_dir)
  json_path = ckpt_leaf / "ppo_network_config.json"

  # Fast path: actor-only load (no env build) if JSON exists.
  if json_path.exists():
    print(f"[BRAX] Found network config JSON: {json_path}")
    try:
      policy_raw = ppo_checkpoint.load_policy(ckpt_leaf, deterministic=deterministic)
      print("[BRAX] Loaded actor via ppo_checkpoint.load_policy")
      # Still return env/env_cfg (for sampling obs vectors).
      if env is None or env_cfg is None:
        env, env_cfg = _build_env(env_name, env_config_path)
      return ckpt_leaf, env, env_cfg, policy_raw, "ppo_checkpoint.load_policy"
    except Exception as e:  # pylint: disable=broad-except
      print(f"[BRAX] load_policy failed; fallback to ppo.train restore. {type(e).__name__}: {e}")
  else:
    print(f"[BRAX] Missing network config JSON; will use ppo.train restore: {json_path}")

  if env is None or env_cfg is None:
    env, env_cfg = _build_env(env_name, env_config_path)
  ppo_params = _get_restore_like_ppo_params()
  ppo_training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params["network_factory"])

  train_fn = functools.partial(
      ppo.train,
      **ppo_training_params,
      network_factory=network_factory,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      progress_fn=None,
  )
  make_inf, params, _ = train_fn(environment=env, eval_env=env, restore_checkpoint_path=ckpt_leaf)
  policy_raw = make_inf(params, deterministic=deterministic)
  print("[BRAX] Loaded actor via ppo.train restore")
  return ckpt_leaf, env, env_cfg, policy_raw, "ppo.train(restore_checkpoint_path=...)"


def _load_tflite_interpreter(tflite_path: epath.Path) -> tf.lite.Interpreter:
  interp = tf.lite.Interpreter(model_path=str(tflite_path))
  interp.allocate_tensors()
  return interp


def _tflite_infer(interp: tf.lite.Interpreter, x: np.ndarray) -> np.ndarray:
  input_details = interp.get_input_details()
  output_details = interp.get_output_details()
  if len(input_details) != 1 or len(output_details) != 1:
    raise ValueError("Expected exactly 1 input and 1 output tensor in TFLite model.")

  in_shape = tuple(int(s) for s in input_details[0]["shape"])
  idx_in = int(input_details[0]["index"])
  idx_out = int(output_details[0]["index"])

  x = np.asarray(x, dtype=np.float32)
  # Allow feeding a vector into a (1, D) model.
  if x.ndim == 1 and len(in_shape) == 2:
    x = x[None, :]
  # Allow feeding (1, D) into a (D,) model.
  elif x.ndim == 2 and len(in_shape) == 1 and x.shape[0] == 1:
    x = x[0]

  if tuple(x.shape) != in_shape:
    raise ValueError(f"TFLite input shape mismatch: expects {in_shape}, got {tuple(x.shape)}")

  interp.set_tensor(idx_in, x)
  interp.invoke()
  y = interp.get_tensor(idx_out)
  return np.asarray(y, dtype=np.float32)


def _sample_state_vecs(
    *,
    env,
    num_samples: int,
    seed: int,
    obs_source: str,
    state_dim: int,
) -> np.ndarray:
  obs_source = str(obs_source or "env_reset").lower()
  if obs_source == "normal":
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal(size=(int(num_samples), int(state_dim)), dtype=np.float32)
  if obs_source == "uniform":
    rng = np.random.default_rng(int(seed))
    return rng.uniform(low=-1.0, high=1.0, size=(int(num_samples), int(state_dim))).astype(np.float32)
  if obs_source != "env_reset":
    raise ValueError("test.obs_source must be one of: env_reset, normal, uniform")
  if env is None:
    raise ValueError("obs_source=env_reset requires env (fallback restore path). Use normal/uniform if using load_policy only.")

  xs = []
  for i in range(int(num_samples)):
    key = jax.random.PRNGKey(int(seed) + i)
    st = env.reset(key)
    v = np.asarray(st.obs["state"], dtype=np.float32).reshape((-1,))
    if v.shape[0] != int(state_dim):
      if v.shape[0] > int(state_dim):
        v = v[: int(state_dim)]
      else:
        v = np.pad(v, (0, int(state_dim) - int(v.shape[0])), mode="constant")
    xs.append(v)
  return np.stack(xs, axis=0).astype(np.float32)


def _brax_infer(policy_raw, state_vec: np.ndarray) -> np.ndarray:
  obs = {"state": jp.asarray(state_vec, dtype=jp.float32)}
  act, _ = policy_raw(obs, jax.random.PRNGKey(0))
  return np.asarray(act, dtype=np.float32).reshape((-1,))


def compare_one_job(
    *,
    name: str,
    ckpt_dir: epath.Path,
    env_name: str,
    env_config_path: str | None,
    tflite_model: epath.Path,
    deterministic: bool,
    num_samples: int,
    seed: int,
    obs_source: str,
    out_summary_json: epath.Path | None,
    print_examples: int,
):
  # Always build env once so we can sample obs vectors consistently.
  env, env_cfg = _build_env(env_name, env_config_path)
  ckpt_leaf, env, _env_cfg, policy_raw, method = _load_brax_policy(
      ckpt_dir=ckpt_dir,
      env_name=env_name,
      env_config_path=env_config_path,
      env=env,
      env_cfg=env_cfg,
      deterministic=deterministic,
  )
  # TFLite setup timing (interpreter creation + allocate).
  t_setup0 = time.perf_counter()
  interp = _load_tflite_interpreter(tflite_model)
  tflite_setup_s = time.perf_counter() - t_setup0

  in_shape = tuple(int(s) for s in interp.get_input_details()[0]["shape"])
  out_shape = tuple(int(s) for s in interp.get_output_details()[0]["shape"])
  state_dim = int(in_shape[-1]) if len(in_shape) == 2 else int(in_shape[0])
  action_dim = int(out_shape[-1]) if len(out_shape) >= 1 else 0

  xs = _sample_state_vecs(env=env, num_samples=num_samples, seed=seed, obs_source=obs_source, state_dim=state_dim)

  # JIT the Brax actor over the whole batch for speed.
  key0 = jax.random.PRNGKey(0)

  def _policy_one(state_vec: jp.ndarray) -> jp.ndarray:
    obs = {"state": state_vec}
    act, _ = policy_raw(obs, key0)
    return act

  policy_batch = jax.jit(jax.vmap(_policy_one))
  xs_jp = jp.asarray(xs, dtype=jp.float32)

  # First call: includes compilation (JIT) cost.
  t0 = time.perf_counter()
  acts_brax_jp = policy_batch(xs_jp)
  acts_brax_jp = jax.block_until_ready(acts_brax_jp)
  brax_jit_s = time.perf_counter() - t0

  # Second call: steady-state runtime (no compilation in typical cases).
  t1 = time.perf_counter()
  acts_brax_jp_2 = policy_batch(xs_jp)
  acts_brax_jp_2 = jax.block_until_ready(acts_brax_jp_2)
  brax_run_s = time.perf_counter() - t1

  # Use steady-state outputs for comparison.
  acts_brax = np.asarray(acts_brax_jp_2, dtype=np.float32)

  # Run TFLite (interpreter is inherently imperative; loop is fine).
  # Warm-up one invoke (some delegates/kernels do one-time initialization).
  t_first0 = time.perf_counter()
  _ = _tflite_infer(interp, xs[0])
  tflite_first_invoke_s = time.perf_counter() - t_first0

  t2 = time.perf_counter()
  acts_tfl = []
  for i in range(int(num_samples)):
    acts_tfl.append(_tflite_infer(interp, xs[i]).reshape((-1,)))
  acts_tfl = np.stack(acts_tfl, axis=0).astype(np.float32)
  tfl_run_s = time.perf_counter() - t2

  if acts_brax.shape != acts_tfl.shape:
    raise ValueError(f"Action batch shape mismatch: brax={acts_brax.shape}, tflite={acts_tfl.shape}")
  if action_dim and acts_brax.shape[-1] != int(action_dim):
    raise ValueError(f"TFLite action_dim mismatch: expected {action_dim}, got {acts_brax.shape[-1]}")

  diff = acts_brax - acts_tfl
  maes = np.mean(np.abs(diff), axis=-1).astype(np.float32)
  maxabs = np.max(np.abs(diff), axis=-1).astype(np.float32)
  examples = []

  for i in range(min(int(print_examples), int(num_samples))):
    examples.append(
        {
            "i": int(i),
            "mae": float(maes[i]),
            "max_abs": float(maxabs[i]),
            "brax": acts_brax[i].reshape((-1,)).tolist(),
            "tflite": acts_tfl[i].reshape((-1,)).tolist(),
        }
    )
  summary = {
      "name": name,
      "env_name": env_name,
      "checkpoint_dir": str(ckpt_dir),
      "checkpoint_leaf_dir": str(ckpt_leaf),
      "brax_load_method": method,
      "tflite_model": str(tflite_model),
      "num_samples": int(num_samples),
      "seed": int(seed),
      "obs_source": str(obs_source),
      "state_dim": int(state_dim),
      "action_dim": int(action_dim),
      "timing": {
          "brax_jit_ms": float(brax_jit_s * 1000.0),
          "brax_batch_ms": float(brax_run_s * 1000.0),
          "brax_per_inference_ms": float((brax_run_s * 1000.0) / max(1, int(num_samples))),
          "tflite_setup_ms": float(tflite_setup_s * 1000.0),
          "tflite_first_invoke_ms": float(tflite_first_invoke_s * 1000.0),
          "tflite_batch_ms": float(tfl_run_s * 1000.0),
          "tflite_per_inference_ms": float((tfl_run_s * 1000.0) / max(1, int(num_samples))),
      },
      "mae_mean": float(np.mean(maes)),
      "mae_min": float(np.min(maes)),
      "mae_max": float(np.max(maes)),
      "maxabs_mean": float(np.mean(maxabs)),
      "maxabs_min": float(np.min(maxabs)),
      "maxabs_max": float(np.max(maxabs)),
      "examples": examples,
  }

  print("\n" + "=" * 80)
  print(f"[JOB] {name}")
  print("=" * 80)
  print(f"- env_name: {env_name}")
  print(f"- ckpt_leaf: {ckpt_leaf}")
  print(f"- brax_load: {method}")
  print(f"- tflite: {tflite_model}")
  print(f"- samples: {num_samples} (obs_source={obs_source}, seed={seed})")
  print(
      "- timing(ms): "
      f"brax_jit={summary['timing']['brax_jit_ms']:.2f}, "
      f"brax_avg={summary['timing']['brax_per_inference_ms']:.3f}, "
      f"tflite_setup={summary['timing']['tflite_setup_ms']:.2f}, "
      f"tflite_first={summary['timing']['tflite_first_invoke_ms']:.2f}, "
      f"tflite_avg={summary['timing']['tflite_per_inference_ms']:.3f}"
  )
  print(f"- MAE   : mean={summary['mae_mean']:.6g} min={summary['mae_min']:.6g} max={summary['mae_max']:.6g}")
  print(f"- MAXABS: mean={summary['maxabs_mean']:.6g} min={summary['maxabs_min']:.6g} max={summary['maxabs_max']:.6g}")

  if out_summary_json is not None:
    out_p = epath.Path(out_summary_json)
    if not out_p.is_absolute():
      out_p = (epath.Path(_THIS_DIR) / out_p).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[WRITE] {out_p}")

  return summary


def main():
  cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_THIS_DIR, "test_policy_convert_config.yaml")
  cfg = _load_yaml(cfg_path)

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
    raise ValueError(f"job_name must be a string, a list, or 'all'. Available: {sorted(list(jobs.keys()))}")
  missing = [n for n in job_names if n not in jobs]
  if missing:
    raise ValueError(f"Unknown job_name(s): {missing}. Available: {sorted(list(jobs.keys()))}")

  test_cfg = cfg.get("test", {}) or {}
  num_samples = int(test_cfg.get("num_samples", 100))
  seed = int(test_cfg.get("seed", 0))
  obs_source = str(test_cfg.get("obs_source", "env_reset"))
  deterministic = bool(test_cfg.get("deterministic", True))
  print_examples = int(test_cfg.get("print_examples", 0))
  continue_on_error = bool(test_cfg.get("continue_on_error", True))

  print(f"Using config: {cfg_path}")
  print(f"Jobs: {job_names}")

  summaries = []
  failures: list[tuple[str, str]] = []

  for name in job_names:
    job = jobs[name]

    ckpt_dir = epath.Path(job["checkpoint_dir"])
    if not ckpt_dir.is_absolute():
      ckpt_dir = (epath.Path(_THIS_DIR) / ckpt_dir).resolve()

    env_name = str(job["env_name"])
    env_config_path = job.get("env_config_path", None)

    tflite_model = epath.Path(job["tflite_model"])
    if not tflite_model.is_absolute():
      tflite_model = (epath.Path(_THIS_DIR) / tflite_model).resolve()

    out_summary_json = job.get("out_summary_json", None)

    try:
      if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {ckpt_dir}")
      if not tflite_model.exists():
        raise FileNotFoundError(f"tflite_model not found: {tflite_model}")

      summaries.append(
          compare_one_job(
              name=name,
              ckpt_dir=ckpt_dir,
              env_name=env_name,
              env_config_path=env_config_path,
              tflite_model=tflite_model,
              deterministic=deterministic,
              num_samples=num_samples,
              seed=seed,
              obs_source=obs_source,
              out_summary_json=out_summary_json,
              print_examples=print_examples,
          )
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
    print("[DONE] Completed with failures:")
    for n, m in failures:
      print(f"- {n}: {m}")
    print("=" * 80)

  # Overall summary (quick view)
  if summaries:
    maes = np.asarray([s["mae_mean"] for s in summaries], dtype=np.float32)
    mx = np.asarray([s["maxabs_max"] for s in summaries], dtype=np.float32)
    print("\n" + "=" * 80)
    print("[SUMMARY] Across successful jobs")
    print("=" * 80)
    print(f"- jobs_ok: {len(summaries)}/{len(job_names)}")
    print(f"- mae_mean: mean={float(np.mean(maes)):.6g} min={float(np.min(maes)):.6g} max={float(np.max(maes)):.6g}")
    print(f"- maxabs_max: mean={float(np.mean(mx)):.6g} min={float(np.min(mx)):.6g} max={float(np.max(mx)):.6g}")


if __name__ == "__main__":
  main()

