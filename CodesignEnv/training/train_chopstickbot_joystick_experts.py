#!/usr/bin/env python3
"""
Train a sweep of Chopstickbot joystick *expert* policies over a leg-length model pool.

Unlike the UniformLeg "single policy across models" approach, this script trains
one policy per model variant (len_0.20m, len_0.30m, ...), saving each to a
separate subfolder:

  checkpoints/ChopstickbotJoystickExperts/<expert_name>/final

Optionally, it can chain initialization so that the final checkpoint of len_i
becomes the restore checkpoint for len_{i+1}.
"""

import os
import sys
import time
import re

import yaml
from etils import epath

# Configure MuJoCo / JAX behaviour *before* importing JAX or any MuJoCo users.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

# Enable JAX persistent compilation cache to speed up re-runs on this machine.
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# training/ -> CodesignEnv/ -> Robot_Hardware_Software_Codesign/
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

from CodesignEnv.training import (  # noqa: E402
    curriculum_chopstickbot_flat_to_rough_wandb as curriculum_trainer_pos,
    curriculum_chopstickbot_flat_to_rough_torque_awared_wandb as curriculum_trainer_torque,
)


def _load_config(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def _len_from_name(name: str) -> float:
  m = re.match(r"len_([0-9]+(?:\.[0-9]+)?)m$", str(name))
  if not m:
    raise ValueError(f"Unexpected model folder name: {name}")
  return float(m.group(1))


def _env_name(control_mode: str, terrain: str, use_alt_imu: bool) -> str:
  control_mode = str(control_mode).lower().strip()
  terrain = str(terrain).lower().strip()
  if terrain not in ("flat", "rough"):
    raise ValueError(f"terrain must be 'flat' or 'rough', got: {terrain!r}")

  if control_mode in ("pos", "position", "position_control"):
    if terrain == "flat":
      return "ChopstickbotJoystickFlatTerrainAlter" if use_alt_imu else "ChopstickbotJoystickFlatTerrain"
    return "ChopstickbotJoystickRoughTerrainAlter" if use_alt_imu else "ChopstickbotJoystickRoughTerrain"

  if control_mode in ("torque", "torque_aware", "torque_awared"):
    # Registry currently only provides *AlterTorqueAwared* variants for Chopstickbot.
    if terrain == "flat":
      return "ChopstickbotJoystickFlatTerrainAlterTorqueAwared"
    return "ChopstickbotJoystickRoughTerrainAlterTorqueAwared"

  raise ValueError(
      "control_mode must be 'position' or 'torque', got: "
      f"{control_mode!r}"
  )


def _scene_xml_name(terrain: str) -> str:
  terrain = str(terrain).lower().strip()
  if terrain == "flat":
    return "scene_joystick_flat_terrain_chopstickbot.xml"
  if terrain == "rough":
    return "scene_joystick_rough_terrain_chopstickbot.xml"
  raise ValueError(f"Unknown terrain: {terrain!r}")


def main():
  config_path = os.path.join(_THIS_DIR, "train_chopstickbot_joystick_experts_config.yaml")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"expert sweep config not found at: {config_path}")

  cfg = _load_config(config_path)

  models_dir = cfg.get("models_dir")
  if not models_dir:
    raise ValueError("Config must set 'models_dir'.")
  models_dir = epath.Path(models_dir).expanduser()
  if not models_dir.is_absolute():
    models_dir = (epath.Path(_PROJECT_ROOT) / models_dir).resolve()
  models_dir = models_dir.resolve()
  if not models_dir.exists():
    raise FileNotFoundError(f"models_dir not found: {models_dir}")

  terrain = str(cfg.get("terrain", "flat")).lower().strip()
  control_mode = str(cfg.get("control_mode", "position")).lower().strip()
  use_alt_imu = bool(cfg.get("use_alternative_imu", True))

  num_timesteps_per_model = int(cfg.get("num_timesteps_per_model", 100_000_000))
  chain_restore = bool(cfg.get("chain_restore", True))
  initial_from_checkpoint = cfg.get("initial_from_checkpoint")
  restore = epath.Path(initial_from_checkpoint).expanduser().resolve() if initial_from_checkpoint else None

  checkpoint_root = epath.Path(str(cfg.get("checkpoint_root", "checkpoints/ChopstickbotJoystickExperts"))).expanduser()
  if not checkpoint_root.is_absolute():
    checkpoint_root = (epath.Path(_THIS_DIR) / checkpoint_root).resolve()
  checkpoint_root = checkpoint_root.resolve()

  checkpoint_tag = str(cfg.get("checkpoint_tag", "") or "").strip()

  env_config_path = cfg.get("env_config_path")

  ppo_cfg = cfg.get("ppo", {}) or {}
  num_evals = int(ppo_cfg.get("num_evals", 10))
  num_envs = int(ppo_cfg.get("num_envs", 8_192))
  num_eval_envs = int(ppo_cfg.get("num_eval_envs", 64))
  save_intermediate = bool(ppo_cfg.get("save_intermediate_checkpoints", False))

  # Set W&B API key if provided.
  wandb_cfg = cfg.get("wandb", {}) or {}
  api_key = wandb_cfg.get("api_key")
  if api_key:
    os.environ["WANDB_API_KEY"] = str(api_key)
  else:
    os.environ.setdefault("WANDB_MODE", "disabled")

  env_name = _env_name(control_mode=control_mode, terrain=terrain, use_alt_imu=use_alt_imu)
  xml_name = _scene_xml_name(terrain)
  curriculum_trainer = (
      curriculum_trainer_torque
      if control_mode in ("torque", "torque_aware", "torque_awared")
      else curriculum_trainer_pos
  )

  model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir() and p.name.startswith("len_")])
  if not model_dirs:
    raise RuntimeError(f"No model subfolders found under: {models_dir}")

  t0 = time.perf_counter()
  print("=" * 80)
  print("[EXPERT_SWEEP] Starting expert sweep")
  print(f"[EXPERT_SWEEP] env_name={env_name}")
  print(f"[EXPERT_SWEEP] trainer={type(curriculum_trainer).__name__}")
  print(f"[EXPERT_SWEEP] terrain={terrain} control_mode={control_mode} use_alt_imu={use_alt_imu}")
  print(f"[EXPERT_SWEEP] models_dir={models_dir}")
  print(f"[EXPERT_SWEEP] xml_name={xml_name}")
  print(f"[EXPERT_SWEEP] checkpoint_root={checkpoint_root}")
  print(f"[EXPERT_SWEEP] num_models={len(model_dirs)} timesteps_per_model={num_timesteps_per_model}")
  print(f"[EXPERT_SWEEP] chain_restore={chain_restore} initial_restore={restore}")
  print("=" * 80)

  for i, md in enumerate(model_dirs):
    leg_L = _len_from_name(md.name)
    xml_override = (md / xml_name).resolve()
    if not xml_override.exists():
      raise FileNotFoundError(f"Expected scene XML not found: {xml_override}")

    expert_name = f"{control_mode}_{md.name}"
    if checkpoint_tag:
      expert_name = f"{control_mode}_{checkpoint_tag}_{md.name}"

    stage_name = f"{terrain}_{md.name}"
    print("\n" + "=" * 80)
    print(
        f"[EXPERT_SWEEP] {i+1}/{len(model_dirs)} model={md.name} "
        f"leg_length_m={leg_L:.3f} expert_name={expert_name}"
    )
    print(f"[EXPERT_SWEEP] xml_override={xml_override}")
    print(f"[EXPERT_SWEEP] restore_checkpoint_path={restore}")
    print("=" * 80)

    final_ckpt_dir = curriculum_trainer.train_stage(
        env_name=str(env_name),
        stage_name=str(stage_name),
        num_timesteps=int(num_timesteps_per_model),
        num_evals=int(num_evals),
        num_envs=int(num_envs),
        num_eval_envs=int(num_eval_envs),
        restore_checkpoint_path=restore,
        env_config_path=env_config_path,
        env_config_overrides={"xml_path_override": xml_override.as_posix()},
        save_intermediate_checkpoints=bool(save_intermediate),
        checkpoint_root=checkpoint_root,
        checkpoint_name=str(expert_name),
        wandb_extra_log_data={
            "expert/model_dir": md.name,
            "expert/leg_length_m": float(leg_L),
            "expert/xml_override": xml_override.as_posix(),
            "expert/control_mode": str(control_mode),
            "expert/terrain": str(terrain),
        },
    )

    if chain_restore:
      restore = epath.Path(final_ckpt_dir)

  elapsed = time.perf_counter() - t0
  print("=" * 80)
  print(f"[EXPERT_SWEEP] Done. Total elapsed: {elapsed:.1f}s")
  print("=" * 80)


if __name__ == "__main__":
  main()

