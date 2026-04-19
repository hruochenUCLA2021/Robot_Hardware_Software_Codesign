#!/usr/bin/env python3
"""
Print box-geom axes for a single MJCF (one file per run).

This script is intentionally standalone (no dependency on auto_phonebot_box_placing.py) so you can:
- point it at `phonebot_general.xml` and see all motor boxes
- point it at `phonebot_v2_auto_boxes.xml` and see the generated boxes

Definitions
-----------
- "X_in_world": box local +X axis expressed in world frame.
- "X_in_base":  box local +X axis expressed in `base_body_name` frame.
  (Same for Y/Z.)
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: PyYAML. Install with `pip install pyyaml`.\n"
        f"Original import error: {e}"
    )


def _resolve_path(p: str, base_dir: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict at top-level: {path}")
    return data


def _parse_floats(s: str, n: int) -> np.ndarray:
    parts = s.strip().split()
    if len(parts) != n:
        raise ValueError(f"Expected {n} floats, got {len(parts)} from: {s!r}")
    return np.array([float(p) for p in parts], dtype=float)


def _fmt_vec(v: np.ndarray, fmt: str) -> str:
    return " ".join(fmt.format(float(x)) for x in np.array(v, dtype=float).tolist())


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product (wxyz)."""
    aw, ax, ay, az = [float(x) for x in a]
    bw, bx, by, bz = [float(x) for x in b]
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Active rotation matrix from quaternion (wxyz)."""
    q = _quat_normalize(q)
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class BodyXform:
    q_world: np.ndarray  # wxyz


def _find_worldbody(mj: ET.Element) -> ET.Element:
    wb = mj.find("worldbody")
    if wb is None:
        raise ValueError("MJCF missing <worldbody>.")
    return wb


def _index_bodies_by_name(worldbody: ET.Element) -> dict[str, ET.Element]:
    out: dict[str, ET.Element] = {}

    def rec(b: ET.Element) -> None:
        name = b.get("name")
        if name:
            out[name] = b
        for c in b.findall("body"):
            rec(c)

    for b in worldbody.findall("body"):
        rec(b)
    return out


def _build_body_world_rots(worldbody: ET.Element) -> dict[str, BodyXform]:
    out: dict[str, BodyXform] = {}

    def rec(body: ET.Element, q_parent: np.ndarray) -> None:
        name = body.get("name")
        if not name:
            return
        q_local = _parse_floats(body.get("quat", "1 0 0 0"), 4)
        q_world = _quat_normalize(_quat_mul(q_parent, q_local))
        out[name] = BodyXform(q_world=q_world)
        for child in body.findall("body"):
            rec(child, q_world)

    for b in worldbody.findall("body"):
        rec(b, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    return out


def _select_box_geoms(body_el: ET.Element, cfg: dict[str, Any]) -> list[tuple[str, ET.Element]]:
    """
    Return list of (label, geom_element) for the body.

    Selection rules:
    1) If `geom_names` in cfg: pick those geoms by exact name.
    2) Else if `prefer_named` and any named box geoms exist: pick named box geoms whose name contains any token in `name_contains`.
    3) Else: pick box geoms with group==group_filter (default "1"), fallback to all box geoms.
    """
    geom_names = cfg.get("geom_names")
    if isinstance(geom_names, list) and geom_names:
        out: list[tuple[str, ET.Element]] = []
        name_to_geom = {g.get("name"): g for g in body_el.findall("geom") if g.get("name")}
        for n in geom_names:
            g = name_to_geom.get(str(n))
            if g is not None:
                out.append((str(n), g))
        return out

    # Heuristic selection.
    box_geoms = [g for g in body_el.findall("geom") if g.get("type") == "box"]
    if not box_geoms:
        return []

    prefer_named = bool(cfg.get("prefer_named", True))
    name_contains = cfg.get("name_contains")
    tokens = [str(t) for t in name_contains] if isinstance(name_contains, list) else []

    if prefer_named:
        named = [g for g in box_geoms if g.get("name")]
        if tokens:
            named = [g for g in named if any(tok in (g.get("name") or "") for tok in tokens)]
        if named:
            return [(g.get("name") or "unnamed", g) for g in named]

    group_filter = str(cfg.get("group_filter", "1"))
    group_boxes = [g for g in box_geoms if (g.get("group") == group_filter)]
    chosen = group_boxes if group_boxes else box_geoms

    out2: list[tuple[str, ET.Element]] = []
    for i, g in enumerate(chosen):
        label = g.get("name") or f"unnamed_box_{i}"
        out2.append((label, g))
    return out2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("print_phonbot_box_geom_axis_config.yaml")),
        help="Path to YAML config.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent
    cfg = _load_yaml(cfg_path)

    mjcf_path = _resolve_path(str(cfg["mjcf"]), cfg_dir)
    base_body_name = str(cfg.get("base_body_name", "base_motor_link"))
    float_fmt = str(cfg.get("float_fmt", "{:.6f}"))
    bodies = cfg.get("bodies")
    if not isinstance(bodies, list) or not bodies:
        raise ValueError("Config must contain a non-empty `bodies` list.")
    body_names = [str(b) for b in bodies]

    geom_sel_cfg = cfg.get("geom_selection", {})
    if not isinstance(geom_sel_cfg, dict):
        raise ValueError("geom_selection must be a dict.")

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    wb = _find_worldbody(root)
    body_rots = _build_body_world_rots(wb)
    body_els = _index_bodies_by_name(wb)

    if base_body_name not in body_rots:
        raise SystemExit(f"Missing base body {base_body_name!r} in {mjcf_path}")
    R_base_w = _quat_to_rot(body_rots[base_body_name].q_world)

    print(f"=== BOX AXES REPORT ===")
    print(f"mjcf: {mjcf_path}")
    print(f"base_body_name: {base_body_name}")

    for bn in body_names:
        if bn not in body_els or bn not in body_rots:
            print(f"\n[{bn}] [WARN] missing body")
            continue

        R_body_w = _quat_to_rot(body_rots[bn].q_world)
        selected = _select_box_geoms(body_els[bn], geom_sel_cfg)
        if not selected:
            print(f"\n[{bn}] [WARN] no box geoms selected")
            continue

        for label, geom_el in selected:
            q_geom = _parse_floats(geom_el.get("quat", "1 0 0 0"), 4)
            R_geom = _quat_to_rot(q_geom)

            R_box_w = R_body_w @ R_geom
            R_box_base = R_base_w.T @ R_box_w

            x_w, y_w, z_w = R_box_w[:, 0], R_box_w[:, 1], R_box_w[:, 2]
            x_b, y_b, z_b = R_box_base[:, 0], R_box_base[:, 1], R_box_base[:, 2]

            print(f"\n[{bn}] box={label}")
            print(f"  geom_quat(wxyz): {_fmt_vec(q_geom, float_fmt)}")
            print(f"  X_in_world: {_fmt_vec(x_w, float_fmt)}")
            print(f"  Y_in_world: {_fmt_vec(y_w, float_fmt)}")
            print(f"  Z_in_world: {_fmt_vec(z_w, float_fmt)}")
            print(f"  X_in_base : {_fmt_vec(x_b, float_fmt)}")
            print(f"  Y_in_base : {_fmt_vec(y_b, float_fmt)}")
            print(f"  Z_in_base : {_fmt_vec(z_b, float_fmt)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

