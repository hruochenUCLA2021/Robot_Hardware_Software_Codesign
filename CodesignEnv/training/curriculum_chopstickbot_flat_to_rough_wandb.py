#!/usr/bin/env python3
"""
Curriculum Learning (with W&B): ChopstickbotJoystickFlatTerrain → ChopstickbotJoystickRoughTerrain

This script implements a two-stage curriculum for ChopstickBot:
1. Train on flat terrain.
2. Fine-tune on rough terrain, initializing from the flat policy.

Both stages use 100_000_000 PPO timesteps and log metrics to Weights & Biases.
"""

import os

# Configure MuJoCo / JAX behaviour *before* importing JAX or any MuJoCo users.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

# Set W&B API key for this script.
# If you want to use W&B, set WANDB_API_KEY in your shell env (preferred),
# or set it in the train_*_config.yaml and let the train script export it.

import sys
import functools
from datetime import datetime
from typing import Optional

import jax
from etils import epath
from ml_collections import config_dict
from flax.training import orbax_utils
from orbax import checkpoint as ocp
import wandb
import yaml

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

# Ensure project root (containing Robot_Hardware_Software_Codesign) is on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# training/ -> CodesignEnv/ -> Robot_Hardware_Software_Codesign/
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

from CodesignEnv import registry as env_registry
from CodesignEnv.configs import hi_randomize


def _configure_jax():
  """Configure JAX / MuJoCo flags (similar to flat/rough trainers)."""
  os.environ.setdefault("MUJOCO_GL", "egl")

  xla_flags = os.environ.get("XLA_FLAGS", "")
  if "--xla_gpu_triton_gemm_any" not in xla_flags:
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags


def create_progress_fn(env_name: str, total_steps: int, start_time: datetime,
                       wandb_run: wandb.sdk.wandb_run.Run | None = None):
  """Create progress printing + (optional) W&B logging function."""

  def progress_fn(num_steps: int, metrics):
    reward = metrics.get("eval/episode_reward", 0.0)
    length = metrics.get("eval/episode_length", 0.0)
    pct = 100.0 * float(num_steps) / float(total_steps) if total_steps > 0 else 0.0
    elapsed = datetime.now() - start_time
    elapsed_min = elapsed.total_seconds() / 60.0
    print(
        f"[{env_name}] {num_steps:,}/{total_steps:,} steps "
        f"({pct:5.2f}%) | elapsed={elapsed_min:6.2f} min | "
        f"Reward={float(reward):.2f}, Length={float(length):.1f}"
    )

    progress_dir = (epath.Path(_THIS_DIR) / "progress").resolve()
    progress_dir.mkdir(parents=True, exist_ok=True)

    # Log everything in metrics to W&B if enabled.
    if wandb_run is not None:
      log_data = {"steps": float(num_steps)}
      for k, v in metrics.items():
        try:
          if isinstance(v, (int, float)):
            log_data[k] = v
          else:
            import numpy as np  # local to avoid circulars
            log_data[k] = float(np.asarray(v))
        except Exception:
          continue
      wandb_run.log(log_data, step=int(num_steps))

  return progress_fn


def create_policy_params_fn(ckpt_path: epath.Path):
  """Create checkpoint saving function for intermediate policies."""

  def policy_params_fn(current_step, make_policy, params):
    del make_policy  # Unused.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

  return policy_params_fn


def get_ppo_params_for_chopstickbot() -> config_dict.ConfigDict:
  """Get PPO hyperparameters, using T1 flat terrain as a reference."""
  try:
    ppo_params = locomotion_params.brax_ppo_config("T1JoystickFlatTerrain")
    # CodesignEnv joystick env exposes obs dict with key: "state" only.
    # Some reference configs use value_obs_key="privileged_state"; override to match.
    if "network_factory" in ppo_params:
      ppo_params["network_factory"]["policy_obs_key"] = "state"
      ppo_params["network_factory"]["value_obs_key"] = "state"
    return ppo_params
  except Exception as e:  # pylint: disable=broad-except
    print(f"Warning: failed to load T1 PPO params ({e}), using defaults.")
    return config_dict.ConfigDict(
        {
            "num_timesteps": 5_000_000,
            "num_evals": 10,
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
            "num_envs": 8_192,
            "batch_size": 1024,
            "num_eval_envs": 64,
            "max_gradient_norm": 0.5,
            "seed": 0,
        }
    )


def train_stage(
    env_name: str,
    stage_name: str,
    num_timesteps: int,
    num_evals: int = 10,
    num_envs: int = 8_192,
    num_eval_envs: int = 64,
    restore_checkpoint_path: Optional[epath.Path] = None,
    env_config_path: str | None = None,
) -> epath.Path:
  """Train a single stage (flat or rough) of the ChopstickBot joystick curriculum."""
  _configure_jax()

  print("=" * 80)
  print(f"CURRICULUM STAGE: {stage_name} | ENV: {env_name}")
  print("=" * 80)

  # Get environment class and default config from local Codesign registry.
  EnvClass, default_config = env_registry.get_environment(env_name)
  env_cfg = default_config()

  # Optional: merge in environment overrides from a YAML config file (for
  # example, a previously saved `env_config.yaml` from a training run). This
  # allows you to exactly reproduce a previous env configuration.
  if env_config_path:
    cfg_path = epath.Path(env_config_path).expanduser()
    if not cfg_path.is_absolute():
      cfg_path = epath.Path(_THIS_DIR) / cfg_path
    cfg_path = cfg_path.resolve()
    try:
      with cfg_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
      if isinstance(loaded, dict):
        env_cfg.update(loaded)
      print(f"[CURRICULUM] Applied env_config overrides from: {cfg_path}")
    except Exception as e:  # pylint: disable=broad-except
      print(f"Warning: failed to load env_config from {cfg_path}: {e}")

  print(
      f"Environment config: ctrl_dt={env_cfg.ctrl_dt}, "
      f"sim_dt={env_cfg.sim_dt}"
  )

  # Separate training and evaluation environments.
  # Explicitly choose the joystick task based on env_name so that the correct
  # scene XML is used for each curriculum stage.
  if env_name.endswith("FlatTerrain"):
    task = "flat_terrain"
  elif env_name.endswith("RoughTerrain"):
    task = "rough_terrain"
  else:
    raise ValueError(f"Unexpected env_name for joystick stage: {env_name}")

  env = EnvClass(task=task, config=env_cfg)
  eval_env = EnvClass(task=task, config=env_cfg)
  print(f"Environment created: {type(env).__name__}")
  print(f"Action size: {env.action_size}")

  # PPO hyperparameters with local overrides.
  ppo_params = get_ppo_params_for_chopstickbot()
  ppo_params.num_timesteps = num_timesteps
  ppo_params.num_evals = num_evals
  ppo_params.num_envs = num_envs
  ppo_params.num_eval_envs = num_eval_envs
  print(f"PPO parameters (with curriculum overrides): {ppo_params}")

  # Initialise a separate W&B run for this stage (optional).
  # If the user is not logged in / does not want W&B, we continue without it.
  wandb_run = None
  try:
    wandb_run = wandb.init(
        project="ChopstickbotJoystickCurriculum",
        name=f"{env_name}_{stage_name}",
        config=dict(ppo_params),
    )
  except Exception as e:  # pylint: disable=broad-except
    print(f"[WANDB] Disabled for this run (wandb.init failed): {e}")

  # Stage-specific checkpoint directory.
  base_ckpt_root = (epath.Path(_THIS_DIR) / "checkpoints").resolve()
  ckpt_path = base_ckpt_root / f"{env_name}_{stage_name}"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Persist the resolved environment config alongside checkpoints for this
  # curriculum stage, so we know exactly which env settings produced the
  # checkpoints in this directory.
  env_cfg_path = ckpt_path / "env_config.yaml"
  try:
    cfg_dict = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else dict(env_cfg)
    with env_cfg_path.open("w", encoding="utf-8") as f:
      yaml.safe_dump(cfg_dict, f, sort_keys=False)
    print(f"Saved environment config to: {env_cfg_path}")
  except Exception as e:  # pylint: disable=broad-except
    print(f"Warning: failed to save env_config.yaml ({e})")

  start_time = datetime.now()
  progress_fn = create_progress_fn(
      env_name + f"_{stage_name}", num_timesteps, start_time, wandb_run
  )
  randomizer = hi_randomize.domain_randomize
  policy_params_fn = create_policy_params_fn(ckpt_path)

  # Network factory handling.
  ppo_training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    del ppo_training_params["network_factory"]

  print(f"restore_checkpoint_path for stage '{stage_name}': {restore_checkpoint_path}")

  make_inference_fn, params, metrics = ppo.train(
      environment=env,
      eval_env=eval_env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      network_factory=network_factory,
      randomization_fn=randomizer,
      progress_fn=progress_fn,
      policy_params_fn=policy_params_fn,
      restore_checkpoint_path=restore_checkpoint_path,
      **ppo_training_params,
  )

  end_time = datetime.now()
  print("\n" + "=" * 80)
  print(f"STAGE '{stage_name}' COMPLETED")
  print("=" * 80)
  print(f"Total training time: {end_time - start_time}")

  # Save final parameters for this stage.
  final_ckpt_dir = ckpt_path / "final"
  final_ckpt_dir.mkdir(parents=True, exist_ok=True)

  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  orbax_checkpointer.save(final_ckpt_dir, params, force=True, save_args=save_args)

  print(f"Final policy for stage '{stage_name}' saved to: {final_ckpt_dir}")
  # Finish W&B run for this stage.
  if wandb_run is not None:
    wandb_run.finish()
  return final_ckpt_dir


def main():
  """Run ChopstickBot joystick flat → rough curriculum with 100M steps each."""
  # Stage 1: flat terrain, train from scratch.
  flat_final_ckpt = train_stage(
      env_name="ChopstickbotJoystickFlatTerrain",
      stage_name="flat",
      num_timesteps=100_000_000,
      restore_checkpoint_path=None,
  )

  # Stage 2: rough terrain, fine-tune from flat stage.
  # Pass the flat stage's final checkpoint directory to PPO.
  train_stage(
      env_name="ChopstickbotJoystickRoughTerrain",
      stage_name="rough",
      num_timesteps=100_000_000,
      restore_checkpoint_path=flat_final_ckpt,
  )


if __name__ == "__main__":
  main()


