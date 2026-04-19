#!/usr/bin/env python3
"""
Visualize + inspect MuJoCo keyframes for the Phonebot joystick scenes.

This follows the same style as the HERMES reference scripts:
  - visualize_hi_keyframe.py
  - check_wheel_keyframe.py

Default behavior:
  - Load both alter_v2 scenes in this folder
  - Apply keyframe "home" (qpos + ctrl), falling back to qpos0 if missing
  - Print joint→qpos and actuator→ctrl tables
  - Render a PNG of the keyframe pose (camera="track" if available)

It also supports an optional "values plot" PNG for qpos/ctrl (bars).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET


# 4x4 opaque gray PNG (very small, valid). Used when scene asset files are missing.
_DUMMY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAFUlEQVQImWNgYGD4z0AEYBxVAAAy"
    "uQK1y8vYpAAAAABJRU5ErkJggg=="
)


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _dummy_png_bytes() -> bytes:
    return base64.b64decode(_DUMMY_PNG_B64)


def _build_assets_dict(scene_path: Path) -> dict[str, bytes]:
    """
    MuJoCo resolves <texture file="..."> / <hfield file="..."> at compile time.
    Some repos don't ship these PNGs. For keyframe inspection we don't care about
    their contents, so we provide a tiny in-memory PNG as fallback.
    """
    assets: dict[str, bytes] = {}
    try:
        xml_text = scene_path.read_text(encoding="utf-8")
        root = ET.fromstring(xml_text)
    except Exception:
        return assets

    for elem in root.iter():
        f = elem.attrib.get("file")
        if not f or (not f.lower().endswith(".png")):
            continue

        abs_path = (scene_path.parent / f).resolve()
        if abs_path.is_file():
            data = abs_path.read_bytes()
        else:
            data = _dummy_png_bytes()

        # MuJoCo VFS keys are effectively basenames; avoid duplicates like
        # "../assets/hfield.png" vs "/abs/.../hfield.png".
        assets[Path(f).name] = data

    return assets


def _load_model(scene_path: Path):
    import mujoco

    assets = _build_assets_dict(scene_path)
    with _pushd(scene_path.parent):
        model = mujoco.MjModel.from_xml_path(str(scene_path), assets=assets)
    return model


def _load_keyframe_qpos_ctrl(model, key_name: str):
    """Return (qpos, ctrl) for a named keyframe, or (None, None) if not found."""
    import mujoco
    import numpy as np

    for key_id in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
        if name == key_name:
            key_qpos = np.asarray(model.key_qpos).reshape(model.nkey, model.nq)
            qpos = np.array(key_qpos[key_id], copy=True)

            ctrl = None
            if model.nu > 0 and model.key_ctrl.size == model.nkey * model.nu:
                key_ctrl = np.asarray(model.key_ctrl).reshape(model.nkey, model.nu)
                ctrl = np.array(key_ctrl[key_id], copy=True)
            return qpos, ctrl

    return None, None


def _print_joint_qpos_table(model, qpos) -> None:
    import mujoco
    import numpy as np

    qpos = np.asarray(qpos, dtype=float).ravel()
    print("\n=== Joint → qpos mapping ===")
    print(f"model.nq = {model.nq}, len(qpos) = {len(qpos)}")
    print(f"model.nv = {model.nv}, model.nu = {model.nu}")
    print("-" * 88)
    print(f"{'jnt_id':>6}  {'name':<34}  {'type':<8}  {'qpos_idx':<12}  values")
    print("-" * 88)

    for jnt_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or f"<unnamed_{jnt_id}>"
        jnt_type = int(model.jnt_type[jnt_id])
        qpos_adr = int(model.jnt_qposadr[jnt_id])

        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            nq_j = 7
            type_str = "free"
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            nq_j = 4
            type_str = "ball"
        elif jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
            nq_j = 1
            type_str = "hinge"
        elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
            nq_j = 1
            type_str = "slide"
        else:
            nq_j = 1
            type_str = f"type{jnt_type}"

        idx_start = qpos_adr
        idx_end = idx_start + nq_j
        vals = qpos[idx_start:idx_end]
        idx_str = f"{idx_start}:{idx_end}"
        print(f"{jnt_id:6d}  {name:<34}  {type_str:<8}  {idx_str:<12}  {vals}")

    print("-" * 88)


def _print_actuator_ctrl_table(model, ctrl) -> None:
    import mujoco
    import numpy as np

    if model.nu <= 0:
        print("\n(no actuators: model.nu == 0)")
        return
    if ctrl is None:
        print("\n(no keyframe ctrl provided)")
        return

    ctrl = np.asarray(ctrl, dtype=float).ravel()
    n = min(int(model.nu), int(ctrl.size))
    print("\n=== Actuator → ctrl mapping ===")
    print(f"model.nu = {model.nu}, len(ctrl) = {ctrl.size}")
    print("-" * 88)
    print(f"{'act_id':>6}  {'name':<40}  {'ctrl':>14}")
    print("-" * 88)
    for a in range(n):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}"
        print(f"{a:6d}  {name:<40}  {float(ctrl[a]):14.6g}")
    if ctrl.size != model.nu:
        print(f"(warning: ctrl length {ctrl.size} != model.nu {model.nu}; showing first {n})")
    print("-" * 88)


def _render_keyframe_png(*, scene_path: Path, model, qpos, ctrl, out_png: Path, camera: str | None, show_contact: bool):
    import mujoco
    import numpy as np
    import mediapy as media

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos, dtype=float).ravel()
    if (model.nu > 0) and (ctrl is not None):
        ctrl_flat = np.asarray(ctrl, dtype=float).ravel()
        if ctrl_flat.size == model.nu:
            data.ctrl[:] = ctrl_flat

    mujoco.mj_forward(model, data)

    scene_option = mujoco.MjvOption()
    if show_contact:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        if camera is not None:
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
        else:
            renderer.update_scene(data, scene_option=scene_option)
    except Exception as e:
        print(f"Warning: failed to use camera {camera!r} ({e}); using default camera.")
        renderer.update_scene(data, scene_option=scene_option)

    rgb = renderer.render()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    media.write_image(out_png.as_posix(), rgb)
    print(f"Wrote {out_png}")


def _plot_values_png(*, scene_path: Path, key_name: str, model, qpos, ctrl, out_png: Path) -> None:
    import mujoco
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build labels similar to rollout_phonebot_joystick loader style.
    qpos = np.asarray(qpos, dtype=float).ravel()
    qpos_labels = ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        qpos_labels.append(jname)

    # ctrl labels are actuator names.
    ctrl_labels = []
    for a in range(model.nu):
        ctrl_labels.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}")

    ctrl_arr = np.asarray(ctrl, dtype=float).ravel() if ctrl is not None else np.zeros((0,), dtype=float)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.5])

    ax0 = fig.add_subplot(gs[0])
    x0 = np.arange(min(len(qpos_labels), qpos.size))
    ax0.bar(x0, qpos[: len(x0)], color="#4C78A8")
    ax0.set_xticks(x0)
    ax0.set_xticklabels(qpos_labels[: len(x0)], rotation=45, ha="right", fontsize=9)
    ax0.set_ylabel("qpos")
    ax0.set_title(f"{scene_path.name} | key='{key_name}'")
    ax0.grid(True, axis="y", alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    x1 = np.arange(min(len(ctrl_labels), ctrl_arr.size))
    ax1.bar(x1, ctrl_arr[: len(x1)], color="#F58518")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(ctrl_labels[: len(x1)], rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("ctrl")
    ax1.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_png}")


def _default_scene_paths(script_dir: Path) -> list[Path]:
    return [
        script_dir / "scene_joystick_flat_terrain_alter_v2.xml",
        script_dir / "scene_joystick_rough_terrain_alter_v2.xml",
    ]


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Visualize + inspect MuJoCo keyframes for one or more scene XMLs.")
    ap.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Scene MJCF path (can repeat). Default: the two alter_v2 scenes next to this script.",
    )
    ap.add_argument("--key", default="home", help="Keyframe name to use (default: home).")
    ap.add_argument("--out-dir", default=str(script_dir), help="Output directory (default: script directory).")
    ap.add_argument("--camera", default="track", help="Camera name for rendering (default: track). Use 'none' to use default.")
    ap.add_argument(
        "--show-contact",
        action="store_true",
        default=True,
        help="Render with contact points/forces enabled (default: true). Use --no-show-contact to disable.",
    )
    ap.add_argument(
        "--no-show-contact",
        dest="show_contact",
        action="store_false",
        help="Disable contact visualization in renders.",
    )
    ap.add_argument(
        "--plot-values",
        action="store_true",
        default=True,
        help="Also write a qpos/ctrl bar-plot PNG (default: true). Use --no-plot-values to disable.",
    )
    ap.add_argument(
        "--no-plot-values",
        dest="plot_values",
        action="store_false",
        help="Disable writing the qpos/ctrl bar-plot PNG.",
    )
    args = ap.parse_args()

    scene_paths = [Path(p).expanduser() for p in (args.scene or [])]
    if not scene_paths:
        scene_paths = _default_scene_paths(script_dir)
    scene_paths = [p if p.is_absolute() else (script_dir / p) for p in scene_paths]

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir

    camera = None if str(args.camera).lower() in ("none", "null", "") else str(args.camera)

    for scene in scene_paths:
        print("\n" + "=" * 80)
        print(f"Scene: {scene}")
        model = _load_model(scene)
        print(f"model.nq={model.nq} model.nv={model.nv} model.nu={model.nu} model.nkey={model.nkey}")

        qpos_key, ctrl_key = _load_keyframe_qpos_ctrl(model, args.key)
        if qpos_key is None:
            print(f"Keyframe '{args.key}' not found; using model.qpos0 instead.")
            qpos_key = model.qpos0
            ctrl_key = None
        else:
            print(f"Using keyframe '{args.key}'. len(qpos)={len(qpos_key)} len(ctrl)={0 if ctrl_key is None else len(ctrl_key)}")

        _print_joint_qpos_table(model, qpos_key)
        _print_actuator_ctrl_table(model, ctrl_key)

        out_png = out_dir / f"keyframe_{args.key}_{scene.stem}_render.png"
        _render_keyframe_png(
            scene_path=scene,
            model=model,
            qpos=qpos_key,
            ctrl=ctrl_key,
            out_png=out_png,
            camera=camera,
            show_contact=bool(args.show_contact),
        )

        if bool(args.plot_values):
            out_plot = out_dir / f"keyframe_{args.key}_{scene.stem}_values.png"
            _plot_values_png(scene_path=scene, key_name=args.key, model=model, qpos=qpos_key, ctrl=ctrl_key, out_png=out_plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

