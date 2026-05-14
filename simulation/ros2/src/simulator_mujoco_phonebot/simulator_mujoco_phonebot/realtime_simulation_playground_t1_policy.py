#!/usr/bin/env python3
"""Real-time MuJoCo CPU sim for MuJoCo-Playground T1 joystick policies."""

from __future__ import annotations

import functools
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node


def _find_workspace_root() -> Path:
    def ok(p: Path) -> bool:
        return (p / "reference").is_dir()

    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        if ok(p):
            return p

    this_file = Path(__file__).resolve()
    for p in (this_file.parent, *this_file.parents):
        if ok(p):
            return p

    return cwd


def _add_reference_to_syspath(ws: Path) -> None:
    ref = (ws / "reference").resolve()
    if ref.is_dir() and str(ref) not in sys.path:
        sys.path.insert(0, str(ref))


@dataclass
class TeleopCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


class KeyboardTeleop:
    def __init__(self, max_vx: float, max_vy: float, max_wz: float) -> None:
        self._max_vx = float(max_vx)
        self._max_vy = float(max_vy)
        self._max_wz = float(max_wz)

        self._lock = threading.Lock()
        self._pressed: set[object] = set()
        self._shift = False
        self._ctrl = False
        self._running = True

        try:
            from pynput import keyboard as pynput_keyboard  # type: ignore

            self._kb = pynput_keyboard
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Keyboard teleop requires `pynput`. Install with: pip install pynput"
            ) from exc

        self._listener = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        kb = self._kb

        def on_press(key: object):
            with self._lock:
                if key in (kb.Key.shift, kb.Key.shift_l, kb.Key.shift_r):
                    self._shift = True
                if key in (kb.Key.ctrl, kb.Key.ctrl_l, kb.Key.ctrl_r):
                    self._ctrl = True
                self._pressed.add(key)
                if key == kb.Key.esc:
                    self._running = False
                    return False
                if key == kb.Key.space:
                    self._pressed.clear()
            return None

        def on_release(key: object):
            with self._lock:
                if key in (kb.Key.shift, kb.Key.shift_l, kb.Key.shift_r):
                    self._shift = False
                if key in (kb.Key.ctrl, kb.Key.ctrl_l, kb.Key.ctrl_r):
                    self._ctrl = False
                self._pressed.discard(key)

        self._listener = kb.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def get_command(self) -> TeleopCommand:
        kb = self._kb
        with self._lock:
            pressed = set(self._pressed)
            shift = bool(self._shift)
            ctrl = bool(self._ctrl)

        if shift:
            s = 1.0
        elif ctrl:
            s = 0.25
        else:
            s = 0.5

        up = kb.Key.up in pressed
        down = kb.Key.down in pressed
        left = kb.Key.left in pressed
        right = kb.Key.right in pressed
        k1 = any(getattr(k, "char", None) == "1" for k in pressed)
        k3 = any(getattr(k, "char", None) == "3" for k in pressed)

        vx = (float(up) - float(down)) * s * self._max_vx
        vy = (float(left) - float(right)) * s * self._max_vy
        wz = (float(k1) - float(k3)) * s * self._max_wz
        return TeleopCommand(vx=vx, vy=vy, wz=wz)

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._pressed.clear()
        try:
            if self._listener is not None:
                self._listener.stop()
        except Exception:
            pass


class RealtimePlaygroundT1PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("realtime_playground_t1_policy")

        ws = _find_workspace_root()
        _add_reference_to_syspath(ws)

        self.declare_parameter("env_name", "T1JoystickFlatTerrain")
        self.declare_parameter("checkpoint_dir", "")
        self.declare_parameter(
            "xml_path",
            str(
                ws
                / "reference"
                / "mujoco_playground"
                / "_src"
                / "locomotion"
                / "t1"
                / "xmls"
                / "scene_mjx_feetonly_flat_terrain.xml"
            ),
        )
        self.declare_parameter("action_scale", 1.0)
        self.declare_parameter("max_vx", 1.0)
        self.declare_parameter("max_vy", 0.5)
        self.declare_parameter("max_wz", 1.5)
        self.declare_parameter("show_contact", False)

        env_name = str(self.get_parameter("env_name").value)
        ckpt = str(self.get_parameter("checkpoint_dir").value).strip()
        if not ckpt:
            raise RuntimeError("checkpoint_dir param is required for Playground T1 policy.")
        ckpt_dir = Path(ckpt).expanduser()
        xml_path = Path(str(self.get_parameter("xml_path").value)).expanduser()
        if not ckpt_dir.is_absolute():
            ckpt_dir = (ws / ckpt_dir).resolve()
        if not xml_path.is_absolute():
            xml_path = (ws / xml_path).resolve()

        import jax  # pylint: disable=import-error
        import jax.numpy as jp  # pylint: disable=import-error
        import mujoco  # pylint: disable=import-error
        import mujoco.viewer  # pylint: disable=import-error
        from etils import epath  # pylint: disable=import-error
        from ml_collections import config_dict  # pylint: disable=import-error
        from brax.training.agents.ppo import networks as ppo_networks  # pylint: disable=import-error
        from brax.training.agents.ppo import train as ppo  # pylint: disable=import-error
        from mujoco_playground import wrapper  # pylint: disable=import-error
        from mujoco_playground.config import locomotion_params  # pylint: disable=import-error
        from mujoco_playground._src import registry as mj_registry  # pylint: disable=import-error

        self._jax = jax
        self._jp = jp
        self._mujoco = mujoco
        self._viewer_mod = mujoco.viewer

        env = mj_registry.load(env_name)

        try:
            ppo_params = locomotion_params.brax_ppo_config(env_name)
        except Exception:
            ppo_params = config_dict.ConfigDict(
                {
                    "num_timesteps": 0,
                    "num_evals": 1,
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

        ppo_params["num_envs"] = 1
        ppo_params["num_eval_envs"] = 1
        if "batch_size" in ppo_params:
            ppo_params["batch_size"] = 1
        ppo_params["num_timesteps"] = 0

        ppo_training_params = dict(ppo_params)
        network_factory = ppo_networks.make_ppo_networks
        if "network_factory" in ppo_params:
            del ppo_training_params["network_factory"]
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **ppo_params["network_factory"]
            )

        train_fn = functools.partial(
            ppo.train, **ppo_training_params, network_factory=network_factory, progress_fn=None
        )
        make_inference_fn, params, _ = train_fn(
            environment=env,
            eval_env=env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            restore_checkpoint_path=epath.Path(str(ckpt_dir)),
        )
        self._policy = jax.jit(make_inference_fn(params, deterministic=True))
        self._rng = jax.random.PRNGKey(0)

        # MuJoCo CPU model.
        self._mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._mj_data = mujoco.MjData(self._mj_model)

        kf = self._mj_model.keyframe("home")
        self._default_qpos = np.array(kf.qpos, dtype=np.float32)
        self._default_pose = np.array(kf.qpos[7:], dtype=np.float32)

        self._mj_data.qpos[:] = self._default_qpos
        self._mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._imu_site_id = int(self._mj_model.site("imu").id)
        self._adr_gyro = int(self._mj_model.sensor("gyro").adr)
        self._adr_local_linvel = int(self._mj_model.sensor("local_linvel").adr)

        self._nu = int(self._mj_model.nu)
        self._last_act = np.zeros((self._nu,), dtype=np.float32)

        # Use env-like ctrl_dt (0.02).
        self._ctrl_dt = 0.02

        self._phase = np.array([0.0, np.pi], dtype=np.float32)
        gait_freq = 1.5
        sim_dt = float(self._mj_model.opt.timestep)
        # IMPORTANT: phase is updated once per *control* step, so use ctrl_dt
        # (not the MuJoCo sim timestep) to match the training env logic.
        self._phase_dt = float(2.0 * np.pi * self._ctrl_dt * gait_freq)

        self._teleop = KeyboardTeleop(
            max_vx=float(self.get_parameter("max_vx").value),
            max_vy=float(self.get_parameter("max_vy").value),
            max_wz=float(self.get_parameter("max_wz").value),
        )
        self._teleop.start()

        self._viewer = self._viewer_mod.launch_passive(
            self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=False
        )
        if bool(self.get_parameter("show_contact").value):
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        self._n_substeps = max(1, int(round(self._ctrl_dt / sim_dt)))
        self.get_logger().info(f"env_name={env_name} ckpt={ckpt_dir}")
        self.get_logger().info(f"xml={xml_path}")
        self.get_logger().info(f"ctrl_dt={self._ctrl_dt:.4f}s n_substeps={self._n_substeps}")

        self._timer = self.create_timer(self._ctrl_dt, self._on_timer)

    def _compute_obs_state(self, cmd: TeleopCommand) -> np.ndarray:
        gyro = np.array(
            self._mj_data.sensordata[self._adr_gyro : self._adr_gyro + 3], dtype=np.float32
        )
        linvel = np.array(
            self._mj_data.sensordata[self._adr_local_linvel : self._adr_local_linvel + 3],
            dtype=np.float32,
        )
        xmat = self._mj_data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity = (xmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

        q = np.array(self._mj_data.qpos[7:], dtype=np.float32)
        qd = np.array(self._mj_data.qvel[6:], dtype=np.float32)
        cmd_arr = np.array([cmd.vx, cmd.vy, cmd.wz], dtype=np.float32)
        phase_feat = np.concatenate([np.cos(self._phase), np.sin(self._phase)]).astype(np.float32)

        # Matches reference `t1/joystick.py` actor state layout.
        state = np.concatenate(
            [
                linvel,
                gyro,
                gravity,
                cmd_arr,
                q - self._default_pose,
                qd,
                self._last_act,
                phase_feat,
            ]
        ).astype(np.float32)
        return state

    def _cleanup(self) -> None:
        try:
            self._teleop.stop()
        except Exception:
            pass
        try:
            if self._viewer is not None:
                self._viewer.close()
        except Exception:
            pass

    def _on_timer(self) -> None:
        try:
            if self._viewer is not None and not self._viewer.is_running():
                self.get_logger().info("Viewer closed; shutting down.")
                self._cleanup()
                rclpy.shutdown()
                return
            if not self._teleop.running:
                self.get_logger().info("Teleop exit requested; shutting down.")
                self._cleanup()
                rclpy.shutdown()
                return

            lock_ctx = self._viewer.lock() if self._viewer is not None else None
            if lock_ctx is None:
                return
            with lock_ctx:
                self._step_once()
        except Exception as e:  # pylint: disable=broad-except
            self.get_logger().error(f"[TIMER] Exception: {type(e).__name__}: {e}")
            self.get_logger().error(traceback.format_exc())
            self._cleanup()
            rclpy.shutdown()

    def _step_once(self) -> None:
        mujoco = self._mujoco
        cmd = self._teleop.get_command()

        cmd_norm = float(np.linalg.norm([cmd.vx, cmd.vy, cmd.wz]))
        if cmd_norm > 0.01:
            phase_tp1 = self._phase + float(self._phase_dt)
            self._phase = (np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi).astype(np.float32)
        else:
            self._phase = np.ones((2,), dtype=np.float32) * np.pi

        obs_state = self._compute_obs_state(cmd)
        obs = {"state": self._jp.asarray(obs_state)}

        act_rng, self._rng = self._jax.random.split(self._rng)
        action = np.array(self._policy(obs, act_rng)[0], dtype=np.float32).reshape((-1,))

        action_scale = float(self.get_parameter("action_scale").value)
        motor_targets = self._default_pose + action * action_scale
        # Soft clip to joint ranges.
        lowers, uppers = self._mj_model.jnt_range[1:].T
        c = (lowers + uppers) / 2.0
        r = (uppers - lowers) * 0.95
        soft_l = c - 0.5 * r
        soft_u = c + 0.5 * r
        motor_targets = np.clip(motor_targets, soft_l, soft_u).astype(np.float32)

        self._mj_data.ctrl[:] = motor_targets
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._mj_model, self._mj_data)

        self._last_act = action
        if self._viewer is not None:
            self._viewer.sync()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealtimePlaygroundT1PolicyNode()
    try:
        from rclpy.executors import MultiThreadedExecutor

        ex = MultiThreadedExecutor(num_threads=2)
        ex.add_node(node)
        ex.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt; cleaning up.")
        node._cleanup()
    finally:
        node.destroy_node()
        rclpy.shutdown()

