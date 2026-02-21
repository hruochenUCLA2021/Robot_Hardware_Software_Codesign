#!/usr/bin/env python3
"""
Unified entry point for training PhoneBot joystick policies (flat, rough, or curriculum).

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
    curriculum_phonebot_flat_to_rough_wandb as curriculum_trainer,
)


def _load_config(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def main():
  # Load training config.
  config_path = os.path.join(_THIS_DIR, "train_phonebot_joystick_config.yaml")
  if not os.path.exists(config_path):
    raise FileNotFoundError(
        f"train config not found at: {config_path}. "
        "Please create it before running train_phonebot_joystick.py."
    )

  cfg = _load_config(config_path)

  # Optional environment config overrides (shared for all joystick modes).
  # If provided, this YAML path is forwarded to the curriculum trainer, which
  # merges it into the base joystick ConfigDict for each stage.
  env_section = cfg.get("environment", {}) or {}
  env_cfg_path = env_section.get("env_config_path")

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

    curriculum_trainer.train_stage(
        env_name="PhonebotJoystickFlatTerrain",
        stage_name="flat",
        num_timesteps=num_timesteps,
        num_evals=int(flat_cfg.get("num_evals", 10)),
        num_envs=int(flat_cfg.get("num_envs", 8_192)),
        num_eval_envs=int(flat_cfg.get("num_eval_envs", 64)),
        restore_checkpoint_path=restore,
        env_config_path=env_cfg_path,
    )

  elif mode == "rough":
    rough_cfg = cfg.get("rough", {}) or {}
    num_timesteps = int(rough_cfg.get("num_timesteps", 100_000_000))
    ckpt = rough_cfg.get("from_checkpoint")
    restore = epath.Path(ckpt) if ckpt else None

    curriculum_trainer.train_stage(
        env_name="PhonebotJoystickRoughTerrain",
        stage_name="rough",
        num_timesteps=num_timesteps,
        num_evals=int(rough_cfg.get("num_evals", 10)),
        num_envs=int(rough_cfg.get("num_envs", 8_192)),
        num_eval_envs=int(rough_cfg.get("num_eval_envs", 64)),
        restore_checkpoint_path=restore,
        env_config_path=env_cfg_path,
    )

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
        env_name="PhonebotJoystickFlatTerrain",
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
        env_name="PhonebotJoystickRoughTerrain",
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


