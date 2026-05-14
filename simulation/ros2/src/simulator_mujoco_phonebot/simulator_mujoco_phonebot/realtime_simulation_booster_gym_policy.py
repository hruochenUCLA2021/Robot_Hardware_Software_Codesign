#!/usr/bin/env python3
"""Real-time MuJoCo CPU sim for BoosterGym T1 TorchScript policy.

This is a MuJoCo-CPU viewer-based runner (like the Phonebot realtime nodes) but
uses BoosterGym's deploy-time policy interface:
`reference/booster_gym/deploy/utils/policy.py`.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node


def _find_workspace_root() -> Path:
    """Best-effort: find a directory that contains `reference/`."""

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


@dataclass
class TeleopCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


class KeyboardTeleop:
    """Keyboard teleop (arrow keys + 1/3 yaw), Esc exits, Space zeros command."""

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


class _BoosterDeployPolicy:
    """BoosterGym *deploy* policy wrapper (TorchScript) for 23-DoF hardware model."""

    def __init__(self, cfg: dict, *, policy_path: str):
        import torch  # pylint: disable=import-error

        self.cfg = cfg
        self.policy = torch.jit.load(str(policy_path))
        self.policy.eval()

        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)
        self.torque_limit = np.array(self.cfg["common"]["torque_limit"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)
        self.gait_frequency_nominal = float(self.cfg["policy"]["gait_frequency"])
        self.gait_frequency = float(self.cfg["policy"]["gait_frequency"])
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(int(self.cfg["policy"]["num_observations"]), dtype=np.float32)
        self.actions = np.zeros(int(self.cfg["policy"]["num_actions"]), dtype=np.float32)

        dt = float(self.cfg["common"]["dt"])
        decim = int(self.cfg["policy"]["control"]["decimation"])
        self.policy_interval = dt * decim

    def get_policy_interval(self) -> float:
        return float(self.policy_interval)

    def inference(
        self,
        time_now: float,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        vx: float,
        vy: float,
        vyaw: float,
    ) -> np.ndarray:
        import torch  # pylint: disable=import-error

        self.gait_process = float(np.fmod(time_now * self.gait_frequency, 1.0))
        self.commands[:] = (vx, vy, vyaw)

        # Smooth commands (clipped slew-rate).
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(
            self.commands - self.smoothed_commands, *clip_range
        )

        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = 0.0
        else:
            self.gait_frequency = self.gait_frequency_nominal

        n = self.cfg["policy"]["normalization"]
        self.obs[0:3] = projected_gravity * float(n["gravity"])
        self.obs[3:6] = base_ang_vel * float(n["ang_vel"])

        moving = float(self.gait_frequency > 1.0e-8)
        self.obs[6] = self.smoothed_commands[0] * float(n["lin_vel"]) * moving
        self.obs[7] = self.smoothed_commands[1] * float(n["lin_vel"]) * moving
        self.obs[8] = self.smoothed_commands[2] * float(n["ang_vel"]) * moving
        self.obs[9] = np.cos(2 * np.pi * self.gait_process) * moving
        self.obs[10] = np.sin(2 * np.pi * self.gait_process) * moving

        # Hardware ordering uses indices 11: for legs.
        self.obs[11:23] = (dof_pos - self.default_dof_pos)[11:] * float(n["dof_pos"])
        self.obs[23:35] = dof_vel[11:] * float(n["dof_vel"])
        self.obs[35:47] = self.actions

        y = self.policy(torch.from_numpy(self.obs).unsqueeze(0))
        self.actions[:] = np.asarray(y.detach().numpy(), dtype=np.float32).reshape(
            (-1,)
        )
        clip = float(n["clip_actions"])
        self.actions[:] = np.clip(self.actions, -clip, clip)

        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += float(self.cfg["policy"]["control"]["action_scale"]) * self.actions
        return self.dof_targets

    def pd_torque(self, q: np.ndarray, qd: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        tau = (q_target - q) * self.stiffness - qd * self.damping
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        return tau.astype(np.float32)


class _BoosterMujocoPolicy:
    """BoosterGym *play_mujoco* style policy wrapper for T1_locomotion.xml (12 actions)."""

    def __init__(self, cfg: dict, *, mj_model, policy_path: str):
        import torch  # pylint: disable=import-error

        self.cfg = cfg
        self.policy = torch.jit.load(str(policy_path))
        self.policy.eval()

        self.nu = int(mj_model.nu)
        self.obs = np.zeros(int(cfg["env"]["num_observations"]), dtype=np.float32)
        self.actions = np.zeros(int(cfg["env"]["num_actions"]), dtype=np.float32)

        self.dt = float(cfg["sim"]["dt"])
        self.decimation = int(cfg["control"]["decimation"])
        self.policy_interval = self.dt * self.decimation

        self.gait_frequency_nominal = float(np.average(cfg["commands"]["gait_frequency"]))
        self.gait_frequency = 0.0
        self.gait_process = 0.0

        # Per-actuator defaults + gains (match play_mujoco.py).
        self.default_dof_pos = np.zeros((self.nu,), dtype=np.float32)
        self.stiffness = np.zeros((self.nu,), dtype=np.float32)
        self.damping = np.zeros((self.nu,), dtype=np.float32)

        default_angles = cfg["init_state"]["default_joint_angles"]
        stiff = cfg["control"]["stiffness"]
        damp = cfg["control"]["damping"]

        for i in range(self.nu):
            name = mj_model.actuator(i).name

            found = False
            for k, v in default_angles.items():
                if k != "default" and (k in name):
                    self.default_dof_pos[i] = float(v)
                    found = True
                    break
            if not found:
                self.default_dof_pos[i] = float(default_angles["default"])

            found = False
            for k in stiff.keys():
                if k in name:
                    self.stiffness[i] = float(stiff[k])
                    self.damping[i] = float(damp[k])
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {name} not defined in cfg['control']")

        self.dof_targets = np.copy(self.default_dof_pos)

    def get_policy_interval(self) -> float:
        return float(self.policy_interval)

    def set_command(self, vx: float, vy: float, vyaw: float) -> None:
        if (vx == 0.0) and (vy == 0.0) and (vyaw == 0.0):
            self.gait_frequency = 0.0
        else:
            self.gait_frequency = self.gait_frequency_nominal
        self.commands = (float(vx), float(vy), float(vyaw))

    def advance_gait(self) -> None:
        self.gait_process = float(
            np.fmod(self.gait_process + self.policy_interval * self.gait_frequency, 1.0)
        )

    def build_obs(
        self,
        *,
        projected_gravity: np.ndarray,
        base_ang_vel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        n = self.cfg["normalization"]
        vx, vy, vyaw = self.commands

        self.obs[0:3] = projected_gravity * float(n["gravity"])
        self.obs[3:6] = base_ang_vel * float(n["ang_vel"])
        self.obs[6] = vx * float(n["lin_vel"])
        self.obs[7] = vy * float(n["lin_vel"])
        self.obs[8] = vyaw * float(n["ang_vel"])

        moving = float(self.gait_frequency > 1.0e-8)
        self.obs[9] = np.cos(2 * np.pi * self.gait_process) * moving
        self.obs[10] = np.sin(2 * np.pi * self.gait_process) * moving

        self.obs[11:23] = (dof_pos - self.default_dof_pos) * float(n["dof_pos"])
        self.obs[23:35] = dof_vel * float(n["dof_vel"])
        self.obs[35:47] = self.actions
        return self.obs

    def inference(self, obs: np.ndarray) -> np.ndarray:
        import torch  # pylint: disable=import-error

        y = self.policy(torch.from_numpy(obs).unsqueeze(0))
        self.actions[:] = np.asarray(y.detach().numpy(), dtype=np.float32).reshape(
            (-1,)
        )
        clip = float(self.cfg["normalization"]["clip_actions"])
        self.actions[:] = np.clip(self.actions, -clip, clip)

        self.dof_targets[:] = self.default_dof_pos + float(self.cfg["control"]["action_scale"]) * self.actions
        return self.dof_targets

    def pd_torque(self, q: np.ndarray, qd: np.ndarray, q_target: np.ndarray, ctrlrange: np.ndarray) -> np.ndarray:
        tau = (q_target - q) * self.stiffness - qd * self.damping
        tau = np.clip(tau, ctrlrange[:, 0], ctrlrange[:, 1])
        return tau.astype(np.float32)


class RealtimeBoosterGymPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("realtime_booster_gym_policy")

        ws = _find_workspace_root()
        self._ws = ws

        # Params
        self.declare_parameter(
            "xml_path",
            str(ws / "reference" / "booster_gym" / "resources" / "T1" / "T1_locomotion.xml"),
        )
        self.declare_parameter(
            "config_path",
            # Default to training/play config (matches BoosterGym play_mujoco.py).
            str(ws / "reference" / "booster_gym" / "envs" / "T1.yaml"),
        )
        self.declare_parameter("policy_path", "")  # optional override
        self.declare_parameter("max_vx", 1.0)
        self.declare_parameter("max_vy", 1.0)
        self.declare_parameter("max_wz", 1.0)
        self.declare_parameter("show_contact", False)

        xml_path = Path(str(self.get_parameter("xml_path").value)).expanduser()
        cfg_path = Path(str(self.get_parameter("config_path").value)).expanduser()
        policy_path = str(self.get_parameter("policy_path").value).strip()

        import mujoco  # pylint: disable=import-error
        import mujoco.viewer  # pylint: disable=import-error
        import yaml  # pylint: disable=import-error

        self._mujoco = mujoco
        self._viewer_mod = mujoco.viewer

        if not xml_path.is_absolute():
            xml_path = (ws / xml_path).resolve()
        if not cfg_path.is_absolute():
            cfg_path = (ws / cfg_path).resolve()

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if policy_path:
            # Allow user-provided relative paths to be interpreted either
            # relative to the deploy config directory (BoosterGym default) OR
            # relative to this repo workspace root (more convenient here).
            pp = Path(policy_path).expanduser()
            if not pp.is_absolute():
                cand_cfg = (cfg_path.parent / pp).resolve()
                cand_ws = (ws / pp).resolve()
                if cand_cfg.exists():
                    policy_path = str(cand_cfg)
                elif cand_ws.exists():
                    policy_path = str(cand_ws)
                else:
                    # Keep the config-relative path for clearer error messages.
                    policy_path = str(cand_cfg)
        else:
            # Resolve cfg policy_path relative to config file.
            rel = str(cfg["policy"]["policy_path"])
            policy_path = str((cfg_path.parent / rel).resolve())

        cfg = dict(cfg)
        self._booster_mode = "deploy" if ("common" in cfg and "policy" in cfg) else "mujoco"

        # Build MuJoCo model first so we can size arrays when using mujoco-mode config.
        self._mj_model = mujoco.MjModel.from_xml_path(str(xml_path))

        if self._booster_mode == "deploy":
            cfg.setdefault("policy", {})
            cfg["policy"] = dict(cfg["policy"])
            cfg["policy"]["policy_path"] = policy_path
            self._policy = _BoosterDeployPolicy(cfg, policy_path=policy_path)
            self._policy_dt = float(self._policy.get_policy_interval())
            self._sim_dt = float(cfg["common"]["dt"])
            self._decimation = int(cfg["policy"]["control"]["decimation"])
            self._mj_model.opt.timestep = self._sim_dt
        else:
            # BoosterGym MuJoCo cross-sim mode (matches reference `play_mujoco.py`).
            self._sim_dt = float(cfg["sim"]["dt"])
            self._decimation = int(cfg["control"]["decimation"])
            self._mj_model.opt.timestep = self._sim_dt
            self._policy = _BoosterMujocoPolicy(cfg, mj_model=self._mj_model, policy_path=policy_path)
            self._policy_dt = float(self._policy.get_policy_interval())

        self.get_logger().info(f"XML: {xml_path}")
        self.get_logger().info(f"Config: {cfg_path}")
        self.get_logger().info(f"Policy: {policy_path}")
        self.get_logger().info(f"Mode: {self._booster_mode}")
        self.get_logger().info(f"Policy interval: {self._policy_dt:.4f}s (decimation={self._decimation})")
        self._mj_data = mujoco.MjData(self._mj_model)

        # Identify IMU site + sensors used by BoosterGym play_mujoco.py.
        try:
            self._imu_site_id = int(self._mj_model.site("imu").id)
        except Exception as e:
            raise RuntimeError("Booster T1 XML must define site name='imu'") from e

        # Gyro (preferred: angular-velocity) and orientation (framequat).
        self._gyro_sensor_adr = None
        try:
            self._gyro_sensor_adr = int(self._mj_model.sensor("angular-velocity").adr)
        except Exception:
            try:
                self._gyro_sensor_adr = int(self._mj_model.sensor("gyro").adr)
            except Exception:
                self._gyro_sensor_adr = None

        try:
            self._orientation_sensor_adr = int(self._mj_model.sensor("orientation").adr)
        except Exception:
            self._orientation_sensor_adr = None

        # Init pose.
        if self._booster_mode == "mujoco":
            # Match BoosterGym `play_mujoco.py` init qpos layout.
            pos = np.array(cfg["init_state"]["pos"], dtype=np.float32)
            rot_xyzw = np.array(cfg["init_state"]["rot"], dtype=np.float32)
            quat_wxyz = np.array([rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]], dtype=np.float32)
            qpos = np.concatenate([pos, quat_wxyz, self._policy.default_dof_pos]).astype(np.float32)
            self._mj_data.qpos[:] = qpos
            self._mj_data.qvel[:] = 0.0
        else:
            qpos = self._mj_data.qpos.copy()
            qpos[7 : 7 + len(self._policy.default_dof_pos)] = self._policy.default_dof_pos
            self._mj_data.qpos[:] = qpos
            self._mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._teleop = KeyboardTeleop(
            max_vx=float(self.get_parameter("max_vx").value),
            max_vy=float(self.get_parameter("max_vy").value),
            max_wz=float(self.get_parameter("max_wz").value),
        )
        self._teleop.start()

        self._t0 = time.perf_counter()
        self._last_policy_time = 0.0

        self._viewer = self._viewer_mod.launch_passive(
            self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=False
        )
        if bool(self.get_parameter("show_contact").value):
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        self._timer = self.create_timer(self._policy_dt, self._on_timer)

    def _projected_gravity(self) -> np.ndarray:
        # World gravity direction in IMU local frame.
        if self._orientation_sensor_adr is not None:
            quat_wxyz = np.array(
                self._mj_data.sensordata[self._orientation_sensor_adr : self._orientation_sensor_adr + 4],
                dtype=np.float32,
            )
            # BoosterGym play_mujoco.py uses x,y,z,w ordering for quat_rotate_inverse.
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            return _quat_rotate_inverse(quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        xmat = self._mj_data.site_xmat[self._imu_site_id].reshape(3, 3)
        return (xmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

    def _base_ang_vel(self) -> np.ndarray:
        if self._gyro_sensor_adr is not None:
            return np.array(
                self._mj_data.sensordata[self._gyro_sensor_adr : self._gyro_sensor_adr + 3],
                dtype=np.float32,
            )
        # Fallback: use base qvel angular velocity (world frame). Not ideal.
        return np.array(self._mj_data.qvel[3:6], dtype=np.float32)

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
        now = time.perf_counter() - self._t0

        dof_pos = np.array(self._mj_data.qpos[7:], dtype=np.float32)
        dof_vel = np.array(self._mj_data.qvel[6:], dtype=np.float32)
        base_ang_vel = self._base_ang_vel()
        proj_g = self._projected_gravity()

        if self._booster_mode == "mujoco":
            self._policy.set_command(float(cmd.vx), float(cmd.vy), float(cmd.wz))
            obs = self._policy.build_obs(
                projected_gravity=proj_g,
                base_ang_vel=base_ang_vel,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
            )
            q_target = self._policy.inference(obs)
            # Match BoosterGym `play_mujoco.py`: hold dof_targets for one control
            # interval, but recompute PD torques at *every* sim step.
            q_target = np.asarray(q_target, dtype=np.float32)
            self._policy.advance_gait()
        else:
            q_target = self._policy.inference(
                time_now=float(now),
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                base_ang_vel=base_ang_vel,
                projected_gravity=proj_g,
                vx=float(cmd.vx),
                vy=float(cmd.vy),
                vyaw=float(cmd.wz),
            )
            q_target = np.asarray(q_target, dtype=np.float32)

        # Step physics for one control interval (decimation substeps).
        for _ in range(self._decimation):
            dof_pos = np.array(self._mj_data.qpos[7:], dtype=np.float32)
            dof_vel = np.array(self._mj_data.qvel[6:], dtype=np.float32)
            if self._booster_mode == "mujoco":
                tau = self._policy.pd_torque(
                    dof_pos, dof_vel, q_target, self._mj_model.actuator_ctrlrange
                )
            else:
                tau = self._policy.pd_torque(dof_pos, dof_vel, q_target)
            self._mj_data.ctrl[:] = tau
            mujoco.mj_step(self._mj_model, self._mj_data)
        if self._viewer is not None:
            self._viewer.sync()


def _quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Match BoosterGym play_mujoco.py `quat_rotate_inverse`."""
    q_w = float(q_xyzw[-1])
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (float(np.dot(q_vec, v)) * 2.0)
    return (a - b + c).astype(np.float32)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealtimeBoosterGymPolicyNode()
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

