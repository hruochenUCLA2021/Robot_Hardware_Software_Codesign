#!/usr/bin/env python3
"""Real-time MuJoCo CPU Phonebot teleop (keyboard -> command -> policy)."""

from __future__ import annotations

import functools
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node


def _add_project_root_to_syspath() -> None:
    """Ensure `Robot_Hardware_Software_Codesign/` is importable."""
    this_file = Path(__file__).resolve()
    # .../simulation/ros2/src/simulator_mujoco_phonebot/
    ros_pkg_root = this_file.parent.parent
    # .../Robot_Hardware_Software_Codesign/
    project_root = ros_pkg_root.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


@dataclass
class TeleopCommand:
    """High-level joystick command in robot local frame."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


class KeyboardTeleop:
    """Keyboard teleop with Shift/Ctrl speed scaling (requires `pynput`)."""

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
                "Keyboard teleop requires `pynput`. "
                "Install with: pip install pynput"
            ) from exc

        self._listener = None

    @property
    def running(self) -> bool:
        """Whether the teleop loop should keep running."""
        return self._running

    def start(self) -> None:
        """Start the background key listener."""
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
        """Return the current command based on pressed keys."""
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

        a = any(getattr(k, "char", None) == "a" for k in pressed)
        d = any(getattr(k, "char", None) == "d" for k in pressed)

        vx = (float(up) - float(down)) * s * self._max_vx
        vy = (float(left) - float(right)) * s * self._max_vy
        wz = (float(a) - float(d)) * s * self._max_wz
        return TeleopCommand(vx=vx, vy=vy, wz=wz)


class PhonebotRealtimeMujocoNode(Node):
    """Run MuJoCo CPU sim and drive a trained joystick policy in real time."""

    def __init__(self) -> None:
        super().__init__("phonebot_realtime_mujoco")

        _add_project_root_to_syspath()

        import jax  # pylint: disable=import-error
        import jax.numpy as jp  # pylint: disable=import-error
        import mujoco  # pylint: disable=import-error
        import mujoco.viewer  # pylint: disable=import-error

        from CodesignEnv import registry as env_registry  # noqa: E402

        self._jax = jax
        self._jp = jp
        self._mujoco = mujoco
        self._viewer_mod = mujoco.viewer
        self._env_registry = env_registry

        self.declare_parameter(
            "env_name", "PhonebotJoystickFlatTerrainAlterFV2TorqueAwared"
        )
        self.declare_parameter("task", "flat_terrain_alternative_imu_fv2_torque")
        self.declare_parameter("checkpoint_dir", "")
        self.declare_parameter("home_keyframe_name", "home")
        self.declare_parameter("render", True)
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("sim_hz", 500.0)
        self.declare_parameter("action_scale", 1.0)
        self.declare_parameter("disable_noise", True)
        self.declare_parameter("disable_push", True)

        self.declare_parameter("max_vx", 1.0)
        self.declare_parameter("max_vy", 0.5)
        self.declare_parameter("max_wz", 1.5)

        env_name = str(self.get_parameter("env_name").value)
        task = str(self.get_parameter("task").value)
        ckpt_dir = str(self.get_parameter("checkpoint_dir").value)
        home_keyframe_name = str(self.get_parameter("home_keyframe_name").value)
        render = bool(self.get_parameter("render").value)

        if not ckpt_dir:
            raise ValueError("ROS param `checkpoint_dir` is required.")

        control_hz = float(self.get_parameter("control_hz").value)
        sim_hz = float(self.get_parameter("sim_hz").value)
        self._ctrl_dt = 1.0 / max(control_hz, 1e-6)
        self._sim_dt = 1.0 / max(sim_hz, 1e-6)
        self._n_substeps = max(int(round(self._ctrl_dt / self._sim_dt)), 1)

        self.get_logger().info(f"env_name: {env_name}")
        self.get_logger().info(f"task: {task}")
        self.get_logger().info(f"checkpoint_dir: {ckpt_dir}")
        self.get_logger().info(f"home_keyframe_name: {home_keyframe_name}")
        self.get_logger().info(
            f"control_hz={control_hz:.1f} sim_hz={sim_hz:.1f} "
            f"substeps={self._n_substeps}"
        )

        # Build MJX env only to restore policy + normalizer.
        env_cls, default_config = self._env_registry.get_environment(env_name)
        env_cfg = default_config()
        env_cfg.home_keyframe_name = home_keyframe_name
        env_cfg.ctrl_dt = float(self._ctrl_dt)
        env_cfg.sim_dt = float(self._sim_dt)
        env_cfg.action_scale = float(self.get_parameter("action_scale").value)

        if bool(self.get_parameter("disable_noise").value):
            try:
                env_cfg.noise_config.level = 0.0
            except Exception:
                pass
            try:
                env_cfg.actuator_noise_config.enable = False
            except Exception:
                pass

        if bool(self.get_parameter("disable_push").value):
            try:
                env_cfg.push_config.enable = False
            except Exception:
                pass

        self._mjx_env = env_cls(task=task, config=env_cfg)
        xml_path = self._mjx_env.xml_path
        self.get_logger().info(f"Using XML: {xml_path}")

        self._policy = self._load_policy(ckpt_dir)

        # MuJoCo CPU sim.
        self._mj_model = self._mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = float(self._sim_dt)
        self._mj_data = self._mujoco.MjData(self._mj_model)

        self._imu_site_id = int(self._mj_model.site("imu").id)

        try:
            qpos0 = np.array(
                self._mj_model.keyframe(home_keyframe_name).qpos, dtype=np.float64
            )
        except KeyError:
            qpos0 = np.array(self._mj_model.qpos0, dtype=np.float64)

        self._mj_data.qpos[:] = qpos0
        self._mj_data.qvel[:] = 0.0
        if self._mj_model.nu > 0:
            self._mj_data.ctrl[:] = 0.0
        self._mujoco.mj_forward(self._mj_model, self._mj_data)

        self._default_pose = qpos0[7:].astype(np.float32)
        self._soft_lowers, self._soft_uppers = self._soft_joint_limits()

        if hasattr(self._mjx_env, "_motor_controller"):
            self._motor_controller = getattr(self._mjx_env, "_motor_controller")
        else:
            raise RuntimeError("Expected torque-aware env to have _motor_controller")

        self._phase_dt = float(2.0 * np.pi * self._ctrl_dt * 1.5)
        self._phase = np.array([0.0, np.pi], dtype=np.float32)
        self._last_act = np.zeros((self._mj_model.nu,), dtype=np.float32)

        self._teleop = KeyboardTeleop(
            max_vx=float(self.get_parameter("max_vx").value),
            max_vy=float(self.get_parameter("max_vy").value),
            max_wz=float(self.get_parameter("max_wz").value),
        )
        self._teleop.start()
        self.get_logger().info(
            "Controls: arrows=vx/vy, a/d=yaw, Shift=fast, Ctrl=slow, "
            "Space=stop, Esc=quit."
        )

        self._viewer = None
        if render:
            self._viewer = self._viewer_mod.launch_passive(
                self._mj_model, self._mj_data
            )

        self._rng = self._jax.random.PRNGKey(0)
        self.create_timer(self._ctrl_dt, self._on_timer)

    def _load_policy(self, ckpt_dir: str):
        """Restore a Brax PPO policy from checkpoint dir."""
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo_train
        from mujoco_playground import wrapper
        from mujoco_playground.config import locomotion_params

        ppo_params = locomotion_params.brax_ppo_config("T1JoystickFlatTerrain")
        if "network_factory" in ppo_params:
            ppo_params["network_factory"]["policy_obs_key"] = "state"
            ppo_params["network_factory"]["value_obs_key"] = "privileged_state"

        ppo_params["num_envs"] = 1
        ppo_params["num_eval_envs"] = 1
        if "batch_size" in ppo_params:
            ppo_params["batch_size"] = 1
        ppo_params["num_timesteps"] = 0

        train_kwargs = dict(ppo_params)
        network_factory = ppo_networks.make_ppo_networks
        if "network_factory" in ppo_params:
            del train_kwargs["network_factory"]
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **ppo_params["network_factory"]
            )

        train_fn = functools.partial(
            ppo_train.train,
            **train_kwargs,
            network_factory=network_factory,
            progress_fn=None,
        )

        make_inf, params, _ = train_fn(
            environment=self._mjx_env,
            eval_env=self._mjx_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            restore_checkpoint_path=str(Path(ckpt_dir).expanduser()),
        )
        policy = self._jax.jit(make_inf(params, deterministic=True))
        return policy

    def _soft_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute soft joint range (match training env)."""
        lowers = self._mj_model.jnt_range[1:, 0]
        uppers = self._mj_model.jnt_range[1:, 1]
        c = (lowers + uppers) / 2.0
        r = uppers - lowers
        soft_lowers = (c - 0.5 * r * 0.95).astype(np.float32)
        soft_uppers = (c + 0.5 * r * 0.95).astype(np.float32)
        return soft_lowers, soft_uppers

    def _compute_state_obs(self, cmd: TeleopCommand) -> np.ndarray:
        """Compute actor obs["state"] (matching the MJX env ordering)."""
        gyro = np.array(self._mj_data.sensor("gyro").data, dtype=np.float32)
        gyro = gyro.reshape((3,))

        xmat = np.array(self._mj_data.site_xmat[self._imu_site_id], dtype=np.float32)
        xmat = xmat.reshape((3, 3))
        gravity = xmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        q = np.array(self._mj_data.qpos[7:], dtype=np.float32)
        qd = np.array(self._mj_data.qvel[6:], dtype=np.float32)

        phase_feat = np.concatenate(
            [np.cos(self._phase), np.sin(self._phase)], axis=0
        ).astype(np.float32)

        cmd_arr = np.array([cmd.vx, cmd.vy, cmd.wz], dtype=np.float32)
        state = np.hstack(
            [
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

    def _on_timer(self) -> None:
        if not self._teleop.running:
            self.get_logger().info("Teleop exit requested; shutting down.")
            rclpy.shutdown()
            return

        cmd = self._teleop.get_command()

        cmd_norm = float(np.linalg.norm([cmd.vx, cmd.vy, cmd.wz]))
        if cmd_norm > 0.01:
            phase_tp1 = self._phase + self._phase_dt
            self._phase = (np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi).astype(
                np.float32
            )
        else:
            self._phase = np.ones((2,), dtype=np.float32) * np.pi

        obs_state = self._compute_state_obs(cmd)
        obs = {
            "state": self._jp.asarray(obs_state),
            "privileged_state": self._jp.asarray(obs_state),
        }

        action = np.array(self._policy(obs, self._rng)[0], dtype=np.float32)
        action = action.reshape((-1,))

        motor_targets = self._default_pose + action * float(
            self.get_parameter("action_scale").value
        )
        motor_targets = np.clip(
            motor_targets, self._soft_lowers, self._soft_uppers
        ).astype(np.float32)

        q = self._jp.asarray(np.array(self._mj_data.qpos[7:], dtype=np.float32))
        qd = self._jp.asarray(np.array(self._mj_data.qvel[6:], dtype=np.float32))
        qdd = self._jp.asarray(np.array(self._mj_data.qacc[6:], dtype=np.float32))
        tau = self._motor_controller.step(q, qd, qdd, self._jp.asarray(motor_targets))
        tau = np.array(tau, dtype=np.float32)

        self._mj_data.ctrl[:] = tau
        for _ in range(self._n_substeps):
            self._mujoco.mj_step(self._mj_model, self._mj_data)

        self._last_act = action
        if self._viewer is not None:
            self._viewer.sync()


def main(args=None) -> None:
    """ROS2 entrypoint."""
    rclpy.init(args=args)
    node = PhonebotRealtimeMujocoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

