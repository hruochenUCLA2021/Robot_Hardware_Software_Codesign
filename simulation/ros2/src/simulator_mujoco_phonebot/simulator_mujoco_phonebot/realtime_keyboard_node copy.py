#!/usr/bin/env python3
"""Real-time MuJoCo CPU Phonebot teleop (keyboard -> command -> policy)."""

from __future__ import annotations

import functools
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
import traceback

import numpy as np
import rclpy
from rclpy.node import Node


def _add_project_root_to_syspath() -> None:
    """Ensure `CodesignEnv` is importable.

    When running via `ros2 run`, this file lives under an *install* prefix, so
    relative-to-`__file__` assumptions often break. We instead search a few
    reasonable locations (cwd + parents) for `Robot_Hardware_Software_Codesign/`.
    """

    def _maybe_add(root: Path) -> bool:
        root = root.resolve()
        if (root / "CodesignEnv").is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return True
        return False

    def _maybe_add_rhsc(parent: Path) -> bool:
        rhsc = (parent / "Robot_Hardware_Software_Codesign").resolve()
        if (rhsc / "CodesignEnv").is_dir():
            if str(rhsc) not in sys.path:
                sys.path.insert(0, str(rhsc))
            return True
        return False

    # 1) If user is running from inside the repo/workspace, cwd search works.
    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if _maybe_add(p) or _maybe_add_rhsc(p):
            return

    # 2) Fallback: try a few parents of this file (may still work in a dev tree).
    this_file = Path(__file__).resolve()
    for p in (this_file.parent, *this_file.parents):
        if _maybe_add(p) or _maybe_add_rhsc(p):
            return

    # If still not found, leave sys.path unchanged and let the import fail with
    # a clear error.


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

        k1 = any(getattr(k, "char", None) == "1" for k in pressed)
        k3 = any(getattr(k, "char", None) == "3" for k in pressed)

        vx = (float(up) - float(down)) * s * self._max_vx
        vy = (float(left) - float(right)) * s * self._max_vy
        wz = (float(k1) - float(k3)) * s * self._max_wz
        return TeleopCommand(vx=vx, vy=vy, wz=wz)

    def stop(self) -> None:
        """Stop the key listener (best-effort)."""
        with self._lock:
            self._running = False
            self._pressed.clear()
        try:
            if self._listener is not None:
                self._listener.stop()
        except Exception:
            pass


class PhonebotRealtimeMujocoNode(Node):
    """Run MuJoCo CPU sim and drive a trained joystick policy in real time."""

    def _resolve_xml_path(self, xml_path: str) -> str:
        """Resolve an XML path from ROS params.

        - Absolute paths are used as-is.
        - Relative paths are resolved relative to the repo root
          (`Robot_Hardware_Software_Codesign/`).
        - As a convenience, we also try resolving relative to the workspace root
          (parent of `Robot_Hardware_Software_Codesign/`) and the current CWD.
        """
        p = Path(str(xml_path)).expanduser()
        if p.is_absolute():
            return str(p.resolve())

        # Find the Robot_Hardware_Software_Codesign/ root via CodesignEnv package.
        try:
            import CodesignEnv as _codesignenv  # type: ignore

            rhsc_root = Path(_codesignenv.__file__).resolve().parent.parent
        except Exception:
            rhsc_root = Path.cwd()

        candidates = [
            (rhsc_root / p).resolve(),
            (rhsc_root.parent / p).resolve(),
            (Path.cwd() / p).resolve(),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        # If nothing exists yet, return the primary resolved path for a clear error.
        return str(candidates[0])

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
        # Optional override: load this scene XML directly in MuJoCo CPU.
        # If empty, use the MJX env's xml_path for the selected env_name/task.
        self.declare_parameter("xml_path", "")
        self.declare_parameter("checkpoint_dir", "")
        self.declare_parameter("policy_format", "auto")  # auto|brax|tflite
        self.declare_parameter("tflite_num_threads", 1)
        self.declare_parameter("tflite_backend", "auto")  # auto|litert|tflite_runtime|tensorflow
        self.declare_parameter("home_keyframe_name", "home")
        self.declare_parameter("render", True)
        self.declare_parameter("show_contact", False)
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
        xml_path_override = str(self.get_parameter("xml_path").value).strip()
        ckpt_dir = str(self.get_parameter("checkpoint_dir").value)
        policy_format = str(self.get_parameter("policy_format").value).lower().strip()
        home_keyframe_name = str(self.get_parameter("home_keyframe_name").value)
        render = bool(self.get_parameter("render").value)
        show_contact = bool(self.get_parameter("show_contact").value)

        if not ckpt_dir:
            raise ValueError("ROS param `checkpoint_dir` is required.")

        control_hz = float(self.get_parameter("control_hz").value)
        sim_hz = float(self.get_parameter("sim_hz").value)
        self._ctrl_dt = 1.0 / max(control_hz, 1e-6)
        self._sim_dt = 1.0 / max(sim_hz, 1e-6)
        self._n_substeps = max(int(round(self._ctrl_dt / self._sim_dt)), 1)

        self.get_logger().info(f"env_name: {env_name}")
        self.get_logger().info(f"task: {task}")
        if xml_path_override:
            self.get_logger().info(f"xml_path override: {xml_path_override}")
        self.get_logger().info(f"checkpoint_dir: {ckpt_dir}")
        self.get_logger().info(f"policy_format: {policy_format}")
        self.get_logger().info(f"home_keyframe_name: {home_keyframe_name}")
        self.get_logger().info(f"show_contact: {show_contact}")
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
        xml_path_env = str(self._mjx_env.xml_path)
        xml_path = xml_path_env
        if xml_path_override:
            xml_path = self._resolve_xml_path(xml_path_override)
            self.get_logger().info(f"Using XML (override): {xml_path}")
            self.get_logger().info(f"MJX env xml_path (for restore/controller): {xml_path_env}")
        else:
            self.get_logger().info(f"Using XML: {xml_path_env}")

        self._policy_backend = "brax"
        self._policy = None
        self._tflite = None

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
            "Controls: arrows=vx/vy, 1/3=yaw, Shift=fast, Ctrl=slow, "
            "Space=stop, Esc=quit."
        )

        self._viewer = None
        if render:
            self._viewer = self._viewer_mod.launch_passive(
                self._mj_model, self._mj_data
            )
            try:
                with self._viewer.lock():
                    self._viewer.opt.flags[
                        self._mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
                    ] = bool(show_contact)
                    self._viewer.opt.flags[
                        self._mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
                    ] = bool(show_contact)
            except Exception:
                # Viewer API can differ across mujoco versions; ignore if unsupported.
                pass

        # Load policy after viewer initialization (helps avoid some native-lib
        # initialization conflicts on certain systems when using TensorFlow).
        self.get_logger().info("[POLICY] Loading policy (may take a while on first run)...")
        self._policy = self._load_policy(policy_path=ckpt_dir, policy_format=policy_format)
        self.get_logger().info(f"[POLICY] Ready (backend={self._policy_backend})")

        self._rng = self._jax.random.PRNGKey(0)
        self.create_timer(self._ctrl_dt, self._on_timer)
        self.get_logger().info("[SIM] Control loop timer started.")

    def _cleanup(self) -> None:
        """Best-effort cleanup for Ctrl+C / shutdown."""
        try:
            if getattr(self, "_teleop", None) is not None:
                self._teleop.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_viewer", None) is not None:
                # Safe to call without lock per MuJoCo docs.
                self._viewer.close()
        except Exception:
            pass

    def _is_orbax_leaf_dir(self, p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        return (p / "_CHECKPOINT_METADATA").exists() or (p / "_METADATA").exists() or (p / "manifest.ocdbt").exists()

    def _resolve_checkpoint_leaf_dir(self, p: Path) -> Path:
        """Accept leaf dir, parent with `final/`, or parent with numeric leaves."""
        p = Path(p).expanduser().resolve()
        if self._is_orbax_leaf_dir(p):
            return p
        if (p / "final").exists() and self._is_orbax_leaf_dir(p / "final"):
            return (p / "final").resolve()
        numeric = []
        try:
            for c in p.iterdir():
                if c.is_dir() and c.name.isdigit() and self._is_orbax_leaf_dir(c):
                    numeric.append((int(c.name), c))
        except Exception:
            numeric = []
        if numeric:
            numeric.sort(key=lambda x: x[0])
            return numeric[-1][1].resolve()
        return p

    def _load_policy(self, *, policy_path: str, policy_format: str):
        """Load policy from either a Brax checkpoint dir or a TFLite model file.

        - Brax: if `ppo_network_config.json` exists, try `ppo_checkpoint.load_policy`
          first, else fallback to `ppo.train(... restore_checkpoint_path=...)`.
        - TFLite: load a TF Lite Interpreter and run inference on obs['state'].
        """
        policy_format = (policy_format or "auto").lower().strip()
        p = Path(policy_path).expanduser()

        # Auto detect: treat *.tflite as TFLite policy.
        if policy_format == "auto":
            if str(p).lower().endswith(".tflite") or p.is_file():
                policy_format = "tflite"
            else:
                policy_format = "brax"

        if policy_format == "tflite":
            model_path = p.resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"TFLite model not found: {model_path}")

            num_threads = int(self.get_parameter("tflite_num_threads").value)
            backend = str(self.get_parameter("tflite_backend").value).lower().strip()

            # Prefer lightweight interpreter to avoid TensorFlow import hangs
            # inside ROS2 + MuJoCo viewer processes.
            interp = None
            last_err = None

            if backend in ("auto", "litert"):
                try:
                    self.get_logger().info("[POLICY][TFLITE] Using LiteRT (ai-edge-litert) interpreter...")
                    from ai_edge_litert.interpreter import Interpreter  # type: ignore

                    interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
                except Exception as e:  # pylint: disable=broad-except
                    last_err = e
                    self.get_logger().warn(
                        "[POLICY][TFLITE] LiteRT import/use failed. "
                        f"{type(e).__name__}: {e}"
                    )
                    if backend == "litert":
                        raise RuntimeError(
                            "Failed to import/use LiteRT. Install with: pip install ai-edge-litert"
                        ) from e

            if interp is None and backend in ("auto", "tflite_runtime"):
                try:
                    self.get_logger().info("[POLICY][TFLITE] Using tflite_runtime interpreter...")
                    from tflite_runtime.interpreter import Interpreter  # type: ignore

                    interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
                except Exception as e:  # pylint: disable=broad-except
                    last_err = e
                    self.get_logger().warn(
                        "[POLICY][TFLITE] tflite_runtime failed. "
                        f"{type(e).__name__}: {e}"
                    )
                    if backend == "tflite_runtime":
                        raise RuntimeError(
                            "Failed to import/use tflite_runtime. Install with: pip install tflite-runtime"
                        ) from e

            if interp is None:
                if backend not in ("auto", "tensorflow"):
                    raise ValueError("tflite_backend must be auto|litert|tflite_runtime|tensorflow")
                # IMPORTANT: TensorFlow import can hang in this ROS2+MuJoCo viewer
                # process on some systems. If auto couldn't use tflite_runtime,
                # fail fast with actionable guidance instead of freezing.
                if backend == "auto":
                    msg = (
                        "tflite_backend=auto could not use LiteRT or tflite_runtime, and falling back to "
                        "TensorFlow is disabled to avoid a known hang at `import tensorflow` in "
                        "this realtime viewer process.\n"
                        "Fix (recommended for Python 3.12): install LiteRT into the same Python env used by `ros2 run`, "
                        "then run with: -p tflite_backend:=litert\n"
                        "Alternative: install tflite-runtime (often not available for Python 3.12) and run with: "
                        "-p tflite_backend:=tflite_runtime\n"
                    )
                    if last_err is not None:
                        msg += f"(tflite_runtime error: {type(last_err).__name__}: {last_err})"
                    raise RuntimeError(msg)
                try:
                    # Avoid TF trying to initialize CUDA/OpenGL interop in the same
                    # process as the MuJoCo viewer. This can hang or crash on some setups.
                    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
                    self.get_logger().info("[POLICY][TFLITE] Importing tensorflow (CPU-only)...")
                    import tensorflow as tf  # pylint: disable=import-error

                    self.get_logger().info(
                        f"[POLICY][TFLITE] TensorFlow imported: {getattr(tf, '__version__', 'unknown')}"
                    )
                    self.get_logger().info(
                        f"[POLICY][TFLITE] Creating TF Lite Interpreter (threads={num_threads})..."
                    )
                    interp = tf.lite.Interpreter(
                        model_path=str(model_path), num_threads=num_threads
                    )
                except Exception as e:  # pylint: disable=broad-except
                    msg = (
                        "Failed to import/use TensorFlow for TFLite inference. "
                        "Try installing tflite-runtime and set -p tflite_backend:=tflite_runtime."
                    )
                    if last_err is not None:
                        msg += f" (tflite_runtime error was: {type(last_err).__name__}: {last_err})"
                    raise RuntimeError(msg) from e

            self.get_logger().info("[POLICY][TFLITE] Allocating tensors...")
            interp.allocate_tensors()
            self.get_logger().info("[POLICY][TFLITE] Interpreter ready.")
            in_details = interp.get_input_details()
            out_details = interp.get_output_details()
            if len(in_details) != 1 or len(out_details) != 1:
                raise ValueError("Expected 1 input + 1 output tensor for TFLite policy.")

            self._tflite = {
                "interp": interp,
                "idx_in": int(in_details[0]["index"]),
                "idx_out": int(out_details[0]["index"]),
                "in_shape": tuple(int(s) for s in in_details[0]["shape"]),
            }
            self._policy_backend = "tflite"
            self.get_logger().info(f"[POLICY] Loaded TFLite model: {model_path}")
            return None

        if policy_format != "brax":
            raise ValueError("policy_format must be auto|brax|tflite")

        # Brax checkpoint loading.
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import checkpoint as ppo_checkpoint
        from brax.training.agents.ppo import train as ppo_train
        from mujoco_playground import wrapper
        from mujoco_playground.config import locomotion_params

        ckpt_leaf = self._resolve_checkpoint_leaf_dir(p)
        json_path = ckpt_leaf / "ppo_network_config.json"
        if json_path.exists():
            try:
                self.get_logger().info(f"[POLICY] Found {json_path}; trying ppo_checkpoint.load_policy")
                policy = ppo_checkpoint.load_policy(str(ckpt_leaf), deterministic=True)
                policy = self._jax.jit(policy)
                self._policy_backend = "brax"
                self.get_logger().info("[POLICY] Loaded via ppo_checkpoint.load_policy")
                return policy
            except Exception as e:  # pylint: disable=broad-except
                self.get_logger().warn(
                    f"[POLICY] ppo_checkpoint.load_policy failed; falling back to ppo.train restore. "
                    f"{type(e).__name__}: {e}"
                )
        else:
            self.get_logger().info(
                f"[POLICY] Missing ppo_network_config.json at {json_path}; using ppo.train restore"
            )

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
            restore_checkpoint_path=str(ckpt_leaf),
        )
        policy = self._jax.jit(make_inf(params, deterministic=True))
        self._policy_backend = "brax"
        self.get_logger().info("[POLICY] Loaded via ppo.train restore")
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

            # If viewer exists, lock while touching mjModel/mjData.
            lock_ctx = self._viewer.lock() if self._viewer is not None else None
            if lock_ctx is None:
                return self._on_timer_unlocked()
            with lock_ctx:
                return self._on_timer_unlocked()
        except Exception as e:  # pylint: disable=broad-except
            self.get_logger().error(f"[TIMER] Exception: {type(e).__name__}: {e}")
            self.get_logger().error(traceback.format_exc())
            self._cleanup()
            rclpy.shutdown()

    def _on_timer_unlocked(self) -> None:
        """Timer logic assuming it's safe to access mjModel/mjData."""
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

        if self._policy_backend == "tflite":
            t = self._tflite
            if t is None:
                raise RuntimeError("TFLite backend selected but interpreter not initialized.")
            x = np.asarray(obs_state, dtype=np.float32)
            in_shape = t["in_shape"]
            if len(in_shape) == 2:
                x = x.reshape((1, -1))
            if tuple(x.shape) != tuple(in_shape):
                raise ValueError(f"TFLite input shape mismatch: expects {in_shape}, got {x.shape}")
            t["interp"].set_tensor(t["idx_in"], x)
            t["interp"].invoke()
            y = t["interp"].get_tensor(t["idx_out"])
            action = np.asarray(y, dtype=np.float32).reshape((-1,))
        else:
            if self._policy is None:
                raise RuntimeError("Brax backend selected but policy is not loaded.")
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
        from rclpy.executors import MultiThreadedExecutor

        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        # Ensure Ctrl+C always triggers cleanup.
        node.get_logger().info("KeyboardInterrupt received; cleaning up.")
        node._cleanup()
    finally:
        node.destroy_node()
        rclpy.shutdown()

