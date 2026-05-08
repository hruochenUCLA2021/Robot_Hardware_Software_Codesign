#!/usr/bin/env python3
"""
Unified entry point for training ChopstickBot joystick policies (flat, rough, or curriculum).

This script reads training configuration from `train_config.yaml` in the same
directory and then dispatches to the appropriate trainer:

- Flat-only training
- Rough-only training
- Flat → Rough curriculum training

It also sets WANDB_API_KEY from the config so you control which account/project
receives the logs (the underlying *_wandb trainers log metrics to W&B).
"""

import os
import sys

import shutil
import time
import yaml
from etils import epath
from ml_collections import config_dict

# Configure MuJoCo / JAX behaviour *before* importing JAX or any MuJoCo users.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

# Enable JAX persistent compilation cache to speed up re-runs on this machine.
# - JAX_COMPILATION_CACHE_DIR: where compiled executables are stored.
# - JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES / *_MIN_COMPILE_TIME_SECS:
#   set to "0" so that *all* compilations are cached, even very small/fast ones.
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# training/ -> CodesignEnv/ -> Robot_Hardware_Software_Codesign/
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

# Reuse the stage trainer from the curriculum W&B script.
from CodesignEnv.training import (  # noqa: E402
    curriculum_chopstickbot_flat_to_rough_wandb as curriculum_trainer,
)


def _load_config(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def _archive_chunk_final_checkpoint(
    *,
    src_final_dir: epath.Path,
    chunk_idx: int,
    model_dir_name: str,
    stage_name: str,
    leg_length_m: float,
    xml_override: epath.Path,
    chunk_steps: int,
):
  """Copies the overwritten `final/` checkpoint into a per-chunk subfolder."""
  if not src_final_dir.exists():
    raise FileNotFoundError(f"Expected final checkpoint dir not found: {src_final_dir}")

  ckpt_root = src_final_dir.parent
  chunk_root = ckpt_root / "chunks" / f"chunk_{chunk_idx:04d}_{model_dir_name}"
  dst_final_dir = chunk_root / "final"
  dst_final_dir.parent.mkdir(parents=True, exist_ok=True)

  # Ensure we capture an exact snapshot for this chunk.
  if dst_final_dir.exists():
    shutil.rmtree(dst_final_dir.as_posix())
  shutil.copytree(src_final_dir.as_posix(), dst_final_dir.as_posix())

  # Also snapshot the env_config used for this chunk (if present).
  env_cfg = ckpt_root / "env_config.yaml"
  if env_cfg.exists():
    shutil.copy2(env_cfg.as_posix(), (chunk_root / "env_config.yaml").as_posix())

  # Small metadata file for debugging / bookkeeping.
  meta = {
      "chunk_idx": int(chunk_idx),
      "stage_name": str(stage_name),
      "model_dir_name": str(model_dir_name),
      "leg_length_m": float(leg_length_m),
      "chunk_steps": int(chunk_steps),
      "xml_override": xml_override.as_posix(),
      "src_final_dir": src_final_dir.as_posix(),
  }
  with (chunk_root / "chunk_meta.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(meta, f, sort_keys=False)


def main():
  # Load training config.
  config_path = os.path.join(_THIS_DIR, "train_chopstickbot_joystick_config_uniform_leg.yaml")
  if not os.path.exists(config_path):
    raise FileNotFoundError(
        f"train config not found at: {config_path}. "
        "Please create it before running train_chopstickbot_joystick_uniform_leg.py."
    )

  cfg = _load_config(config_path)

  # Optional environment config overrides (shared for all joystick modes).
  # If provided, this YAML path is forwarded to the curriculum trainer, which
  # merges it into the base joystick ConfigDict for each stage.
  env_section = cfg.get("environment", {}) or {}
  env_cfg_path = env_section.get("env_config_path")
  use_alt_imu = bool(env_section.get("use_alternative_imu", True))
  env_name_flat = env_section.get("env_name_flat")
  env_name_rough = env_section.get("env_name_rough")
  if not env_name_flat:
    env_name_flat = "ChopstickbotJoystickFlatTerrainAlterUniformLeg" if use_alt_imu else "ChopstickbotJoystickFlatTerrain"
  if not env_name_rough:
    env_name_rough = "ChopstickbotJoystickRoughTerrainAlterUniformLeg" if use_alt_imu else "ChopstickbotJoystickRoughTerrain"

  # Morphology switching pool (Option A).
  uniform_cfg = cfg.get("uniform_leg", {}) or {}
  models_dir = uniform_cfg.get("models_dir")
  switch_chunk = int(uniform_cfg.get("switch_chunk_timesteps", 5_000_000))
  switch_mode = str(uniform_cfg.get("switch_mode", "round_robin")).lower()
  archive_final_per_chunk = bool(uniform_cfg.get("archive_final_per_chunk", True))
  if models_dir:
    # Resolve models_dir relative to project root.
    models_dir = epath.Path(models_dir)
    if not models_dir.is_absolute():
      models_dir = epath.Path(_PROJECT_ROOT) / models_dir
    models_dir = models_dir.resolve()
    if not models_dir.exists():
      raise FileNotFoundError(f"uniform_leg.models_dir not found: {models_dir}")

  # Set W&B API key if provided.
  wandb_cfg = cfg.get("wandb", {}) or {}
  api_key = wandb_cfg.get("api_key")
  if api_key:
    os.environ["WANDB_API_KEY"] = str(api_key)
  else:
    # Default: disable W&B unless the user explicitly provides a key or enables it.
    os.environ.setdefault("WANDB_MODE", "disabled")

  mode = str(cfg.get("mode", "flat")).lower()

  if mode == "flat":
    flat_cfg = cfg.get("flat", {}) or {}
    num_timesteps = int(flat_cfg.get("num_timesteps", 100_000_000))
    ckpt = flat_cfg.get("from_checkpoint")
    restore = epath.Path(ckpt) if ckpt else None

    if not models_dir:
      curriculum_trainer.train_stage(
          env_name=str(env_name_flat),
          stage_name="flat",
          num_timesteps=num_timesteps,
          num_evals=int(flat_cfg.get("num_evals", 10)),
          num_envs=int(flat_cfg.get("num_envs", 8_192)),
          num_eval_envs=int(flat_cfg.get("num_eval_envs", 64)),
          restore_checkpoint_path=restore,
          env_config_path=env_cfg_path,
      )
    else:
      # Option A: switch which XML is used every `switch_chunk` timesteps by
      # running PPO in short chunks and restoring from the previous chunk.
      import re
      import tempfile
      import datetime

      model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir() and p.name.startswith("len_")])
      if not model_dirs:
        raise RuntimeError(f"No model subfolders found under: {models_dir}")
      # Derive leg length from folder name "len_0.50m".
      def _len_from_name(name: str) -> float:
        m = re.match(r"len_([0-9]+(?:\.[0-9]+)?)m$", name)
        if not m:
          raise ValueError(f"Unexpected model folder name: {name}")
        return float(m.group(1))

      n_chunks = max(int((num_timesteps + switch_chunk - 1) // switch_chunk), 1)
      ckpt_leaf = restore
      seen_models: set[str] = set()
      cumulative_steps = 0

      # Create one W&B run for the entire multi-chunk training (if enabled).
      wandb_run = None
      if os.environ.get("WANDB_MODE", "").lower() != "disabled":
        try:
          import wandb  # local import

          wandb_run = wandb.init(
              project="ChopstickbotJoystickCurriculum",
              name=f"{env_name_flat}_flat_uniform_leg_pool_switch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
              config={
                  "env_name": str(env_name_flat),
                  "mode": "flat",
                  "num_timesteps_total": int(num_timesteps),
                  "switch_chunk_timesteps": int(switch_chunk),
                  "n_models": int(len(model_dirs)),
                  "switch_mode": str(switch_mode),
              },
          )
        except Exception as e:  # pylint: disable=broad-except
          print(f"[WANDB] Disabled for this run (wandb.init failed): {e}")
          wandb_run = None
      for i in range(n_chunks):
        md = model_dirs[i % len(model_dirs)] if switch_mode == "round_robin" else model_dirs[int(i % len(model_dirs))]
        leg_L = _len_from_name(md.name)
        xml_override = md / "scene_joystick_flat_terrain_chopstickbot.xml"
        if not xml_override.exists():
          raise FileNotFoundError(f"Expected scene XML not found: {xml_override}")
        overrides = {
            "xml_path_override": xml_override.as_posix(),
            "morphology": {"leg_length_m": leg_L},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
          yaml.safe_dump(overrides, tf, sort_keys=False)
          tmp_path = tf.name

        first_seen = md.name not in seen_models
        seen_models.add(md.name)
        chunk_steps = min(switch_chunk, num_timesteps - i * switch_chunk)
        t0 = time.perf_counter()
        print(
            "\n"
            + "=" * 80
            + f"\n[UNIFORM_LEG] chunk={i+1}/{n_chunks} steps={chunk_steps} "
            f"model={md.name} leg_length_m={leg_L:.3f} first_seen={first_seen}\n"
            + "=" * 80
        )
        ckpt_leaf = curriculum_trainer.train_stage(
            env_name=str(env_name_flat),
            stage_name="flat",
            num_timesteps=chunk_steps,
            num_evals=int(flat_cfg.get("num_evals", 1)),
            num_envs=int(flat_cfg.get("num_envs", 8_192)),
            num_eval_envs=int(flat_cfg.get("num_eval_envs", 64)),
            restore_checkpoint_path=ckpt_leaf,
            env_config_path=tmp_path,
            save_intermediate_checkpoints=False,
            wandb_run=wandb_run,
            step_offset=int(cumulative_steps),
            total_steps_override=int(num_timesteps),
            wandb_extra_log_data={
                "uniform_leg/chunk": int(i + 1),
                "uniform_leg/model_dir": md.name,
                "uniform_leg/leg_length_m": float(leg_L),
                "uniform_leg/first_seen": float(first_seen),
            },
        )
        if archive_final_per_chunk:
          _archive_chunk_final_checkpoint(
              src_final_dir=epath.Path(ckpt_leaf),
              chunk_idx=i + 1,
              model_dir_name=md.name,
              stage_name="flat",
              leg_length_m=leg_L,
              xml_override=xml_override,
              chunk_steps=chunk_steps,
          )
        t1 = time.perf_counter()
        print(
            f"[UNIFORM_LEG] chunk done in {t1 - t0:.2f}s | "
            f"model={md.name} leg_length_m={leg_L:.3f} | ckpt={ckpt_leaf}"
        )
        cumulative_steps += int(chunk_steps)
      if wandb_run is not None:
        try:
          wandb_run.finish()
        except Exception:
          pass

  elif mode == "rough":
    rough_cfg = cfg.get("rough", {}) or {}
    num_timesteps = int(rough_cfg.get("num_timesteps", 100_000_000))
    ckpt = rough_cfg.get("from_checkpoint")
    restore = epath.Path(ckpt) if ckpt else None

    if not models_dir:
      curriculum_trainer.train_stage(
          env_name=str(env_name_rough),
          stage_name="rough",
          num_timesteps=num_timesteps,
          num_evals=int(rough_cfg.get("num_evals", 10)),
          num_envs=int(rough_cfg.get("num_envs", 8_192)),
          num_eval_envs=int(rough_cfg.get("num_eval_envs", 64)),
          restore_checkpoint_path=restore,
          env_config_path=env_cfg_path,
      )
    else:
      import re
      import tempfile
      import datetime

      model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir() and p.name.startswith("len_")])
      if not model_dirs:
        raise RuntimeError(f"No model subfolders found under: {models_dir}")
      def _len_from_name(name: str) -> float:
        m = re.match(r"len_([0-9]+(?:\.[0-9]+)?)m$", name)
        if not m:
          raise ValueError(f"Unexpected model folder name: {name}")
        return float(m.group(1))

      n_chunks = max(int((num_timesteps + switch_chunk - 1) // switch_chunk), 1)
      ckpt_leaf = restore
      seen_models: set[str] = set()
      cumulative_steps = 0

      wandb_run = None
      if os.environ.get("WANDB_MODE", "").lower() != "disabled":
        try:
          import wandb  # local import

          wandb_run = wandb.init(
              project="ChopstickbotJoystickCurriculum",
              name=f"{env_name_rough}_rough_uniform_leg_pool_switch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
              config={
                  "env_name": str(env_name_rough),
                  "mode": "rough",
                  "num_timesteps_total": int(num_timesteps),
                  "switch_chunk_timesteps": int(switch_chunk),
                  "n_models": int(len(model_dirs)),
                  "switch_mode": str(switch_mode),
              },
          )
        except Exception as e:  # pylint: disable=broad-except
          print(f"[WANDB] Disabled for this run (wandb.init failed): {e}")
          wandb_run = None
      for i in range(n_chunks):
        md = model_dirs[i % len(model_dirs)] if switch_mode == "round_robin" else model_dirs[int(i % len(model_dirs))]
        leg_L = _len_from_name(md.name)
        xml_override = md / "scene_joystick_rough_terrain_chopstickbot.xml"
        if not xml_override.exists():
          raise FileNotFoundError(f"Expected scene XML not found: {xml_override}")
        overrides = {
            "xml_path_override": xml_override.as_posix(),
            "morphology": {"leg_length_m": leg_L},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
          yaml.safe_dump(overrides, tf, sort_keys=False)
          tmp_path = tf.name

        first_seen = md.name not in seen_models
        seen_models.add(md.name)
        chunk_steps = min(switch_chunk, num_timesteps - i * switch_chunk)
        t0 = time.perf_counter()
        print(
            "\n"
            + "=" * 80
            + f"\n[UNIFORM_LEG] chunk={i+1}/{n_chunks} steps={chunk_steps} "
            f"model={md.name} leg_length_m={leg_L:.3f} first_seen={first_seen}\n"
            + "=" * 80
        )
        ckpt_leaf = curriculum_trainer.train_stage(
            env_name=str(env_name_rough),
            stage_name="rough",
            num_timesteps=chunk_steps,
            num_evals=int(rough_cfg.get("num_evals", 1)),
            num_envs=int(rough_cfg.get("num_envs", 8_192)),
            num_eval_envs=int(rough_cfg.get("num_eval_envs", 64)),
            restore_checkpoint_path=ckpt_leaf,
            env_config_path=tmp_path,
            save_intermediate_checkpoints=False,
            wandb_run=wandb_run,
            step_offset=int(cumulative_steps),
            total_steps_override=int(num_timesteps),
            wandb_extra_log_data={
                "uniform_leg/chunk": int(i + 1),
                "uniform_leg/model_dir": md.name,
                "uniform_leg/leg_length_m": float(leg_L),
                "uniform_leg/first_seen": float(first_seen),
            },
        )
        if archive_final_per_chunk:
          _archive_chunk_final_checkpoint(
              src_final_dir=epath.Path(ckpt_leaf),
              chunk_idx=i + 1,
              model_dir_name=md.name,
              stage_name="rough",
              leg_length_m=leg_L,
              xml_override=xml_override,
              chunk_steps=chunk_steps,
          )
        t1 = time.perf_counter()
        print(
            f"[UNIFORM_LEG] chunk done in {t1 - t0:.2f}s | "
            f"model={md.name} leg_length_m={leg_L:.3f} | ckpt={ckpt_leaf}"
        )
        cumulative_steps += int(chunk_steps)
      if wandb_run is not None:
        try:
          wandb_run.finish()
        except Exception:
          pass

  elif mode == "curriculum":
    cur_cfg = cfg.get("curriculum", {}) or {}

    flat_num = int(cur_cfg.get("flat_num_timesteps", 100_000_000))
    flat_evals = int(cur_cfg.get("flat_num_evals", 10))
    flat_envs = int(cur_cfg.get("flat_num_envs", 8_192))
    flat_eval_envs = int(cur_cfg.get("flat_num_eval_envs", 64))
    flat_from_ckpt = cur_cfg.get("flat_from_checkpoint")
    flat_restore = epath.Path(flat_from_ckpt) if flat_from_ckpt else None

    # Stage 1: flat terrain.
    flat_final_ckpt = curriculum_trainer.train_stage(
        env_name=str(env_name_flat),
        stage_name="flat",
        num_timesteps=flat_num,
        num_evals=flat_evals,
        num_envs=flat_envs,
        num_eval_envs=flat_eval_envs,
        restore_checkpoint_path=flat_restore,
        env_config_path=env_cfg_path,
    )

    rough_num = int(cur_cfg.get("rough_num_timesteps", 100_000_000))
    rough_evals = int(cur_cfg.get("rough_num_evals", 10))
    rough_envs = int(cur_cfg.get("rough_num_envs", 8_192))
    rough_eval_envs = int(cur_cfg.get("rough_num_eval_envs", 64))
    rough_from_ckpt = cur_cfg.get("rough_from_checkpoint")
    if rough_from_ckpt:
      rough_restore = epath.Path(rough_from_ckpt)
    else:
      # Default: continue from the flat stage's final checkpoint.
      rough_restore = flat_final_ckpt

    # Stage 2: rough terrain.
    curriculum_trainer.train_stage(
        env_name=str(env_name_rough),
        stage_name="rough",
        num_timesteps=rough_num,
        num_evals=rough_evals,
        num_envs=rough_envs,
        num_eval_envs=rough_eval_envs,
        restore_checkpoint_path=rough_restore,
        env_config_path=env_cfg_path,
    )

  else:
    raise ValueError(f"Unknown mode in train_config.yaml: {mode!r}. "
                     "Expected 'flat', 'rough', or 'curriculum'.")


if __name__ == "__main__":
  main()


