#!/usr/bin/env python3
"""
Auto-place Phonebot motor box geoms (pos/quat) to match the reference motor-frame orientation.

Goal
----
In `phonebot_general.xml` the motor body frames are defined in a consistent "motor frame"
(z = output shaft, etc.), so the green debug motor box can use the same pos/quat everywhere.

In newer URDF-converted MJCFs (e.g. `phonebot_v2.xml`), the motor body frames may be rotated
compared to the original convention. This script computes, per motor body, the relative rotation
between the *reference* body frame and the *target* body frame (at zero joint angles), then writes
that as the motor box geom `quat`.

It also maps the reference box offset `pos` (defined in the reference motor body frame) into the
target motor body frame so the **world-space** offset from body origin to box origin matches:

    p_tgt = R_tgt,w^T * R_ref,w * p_ref

Notes on conventions
--------------------
- MuJoCo quaternions are `w x y z`.
- `body quat` is orientation of the body frame relative to its parent.
- World orientation accumulates by quaternion multiplication:
    q_world(child) = q_world(parent) ⊗ q_local(child)

Optional base-frame reconciliation
---------------------------------
Some model variants use different "base frame" axis conventions (e.g. old: y-forward, new: x-forward).
If you want box orientations to be comparable under the *same* base axes, set
`motor_box.reference_rel_quat_premul` in the YAML to a constant quaternion (wxyz) that maps vectors
from the reference base convention into the target base convention. This quaternion is *left-multiplied*
onto the reference relative rotation before computing the target geom quaternion.
"""

from __future__ import annotations

import argparse
import shutil
import sys
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


def _parse_floats(s: str, n: int) -> np.ndarray:
    parts = s.strip().split()
    if len(parts) != n:
        raise ValueError(f"Expected {n} floats, got {len(parts)} from: {s!r}")
    return np.array([float(p) for p in parts], dtype=float)


def _fmt_vec(v: np.ndarray, fmt: str) -> str:
    return " ".join(fmt.format(float(x)) for x in v.tolist())


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def _quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


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


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_to_rot(q) @ np.array(v, dtype=float)


def _map_vec_between_body_world_frames(q_src_body_w: np.ndarray, q_dst_body_w: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    """
    Map a vector expressed in `src` body coordinates into `dst` body coordinates while preserving
    the world-frame direction/magnitude:

        R_dst,w * v_dst = R_src,w * v_src  =>  v_dst = R_dst,w^T * R_src,w * v_src
    """
    q_src_body_w = _quat_normalize(np.array(q_src_body_w, dtype=float))
    q_dst_body_w = _quat_normalize(np.array(q_dst_body_w, dtype=float))
    v_src = np.array(v_src, dtype=float)
    v_w = _quat_rotate(q_src_body_w, v_src)
    return _quat_rotate(_quat_conj(q_dst_body_w), v_w)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (wxyz)."""
    R = np.array(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Expected (3,3) rotation matrix, got {R.shape}")

    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # Find the major diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return _quat_normalize(np.array([w, x, y, z], dtype=float))


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.array(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _twist_quat_about_body_z_to_match_ref(
    *,
    q_ref_rel: np.ndarray,
    q_tgt_rel: np.ndarray,
) -> np.ndarray:
    """
    Compute a quaternion (wxyz) representing a *twist about the target body's local Z axis*
    such that the target box X/Y axes align as closely as possible with the reference box X/Y axes,
    while keeping Z (output shaft) aligned.

    We do this in the base frame (rel rotations), using projected X axis.
    """
    R_ref = _quat_to_rot(q_ref_rel)
    R_tgt = _quat_to_rot(q_tgt_rel)

    z = _unit(R_tgt @ np.array([0.0, 0.0, 1.0], dtype=float))
    x_tgt = _unit(R_tgt @ np.array([1.0, 0.0, 0.0], dtype=float))
    x_ref = _unit(R_ref @ np.array([1.0, 0.0, 0.0], dtype=float))

    # Project reference axis into plane orthogonal to target z.
    x_ref_proj = x_ref - z * float(np.dot(z, x_ref))
    if float(np.linalg.norm(x_ref_proj)) < 1e-9:
        # If ref x is parallel to z, use ref y instead.
        y_ref = _unit(R_ref @ np.array([0.0, 1.0, 0.0], dtype=float))
        x_ref_proj = y_ref - z * float(np.dot(z, y_ref))
    x_ref_proj = _unit(x_ref_proj)

    # Signed angle around z from x_tgt -> x_ref_proj.
    c = float(np.dot(x_tgt, x_ref_proj))
    s = float(np.dot(z, np.cross(x_tgt, x_ref_proj)))
    ang = float(np.arctan2(s, c))

    # Rotation about local Z in *body coordinates* is same angle about z.
    return _quat_normalize(np.array([np.cos(ang / 2.0), 0.0, 0.0, np.sin(ang / 2.0)], dtype=float))


def _twist_quat_about_base_z_to_match_ref(
    *,
    q_ref_rel: np.ndarray,
    q_tgt_rel: np.ndarray,
) -> np.ndarray:
    """
    Compute a quaternion (wxyz) for the *target geom* such that the resulting box orientation
    differs from the target body orientation only by a yaw about the BASE frame +Z axis.

    This is the right choice when you want to reconcile different base-axis conventions
    (e.g. ref uses y-forward, tgt uses x-forward) without introducing any extra tilt in world.

    In base coordinates:
      R_box_base = Rz(ang) * R_tgt_rel
      => R_geom   = R_tgt_rel^T * Rz(ang) * R_tgt_rel
      => q_geom   = q_tgt_rel^{-1} ⊗ qz(ang) ⊗ q_tgt_rel
    """
    R_ref = _quat_to_rot(q_ref_rel)
    R_tgt = _quat_to_rot(q_tgt_rel)

    z_base = np.array([0.0, 0.0, 1.0], dtype=float)
    x_tgt = _unit(R_tgt @ np.array([1.0, 0.0, 0.0], dtype=float))
    x_ref = _unit(R_ref @ np.array([1.0, 0.0, 0.0], dtype=float))

    # Project both x axes into the base XY plane.
    x_tgt_proj = x_tgt - z_base * float(np.dot(z_base, x_tgt))
    x_ref_proj = x_ref - z_base * float(np.dot(z_base, x_ref))
    if float(np.linalg.norm(x_tgt_proj)) < 1e-9 or float(np.linalg.norm(x_ref_proj)) < 1e-9:
        # Fallback: use y axes if x is parallel to base z.
        y_tgt = _unit(R_tgt @ np.array([0.0, 1.0, 0.0], dtype=float))
        y_ref = _unit(R_ref @ np.array([0.0, 1.0, 0.0], dtype=float))
        x_tgt_proj = y_tgt - z_base * float(np.dot(z_base, y_tgt))
        x_ref_proj = y_ref - z_base * float(np.dot(z_base, y_ref))
    x_tgt_proj = _unit(x_tgt_proj)
    x_ref_proj = _unit(x_ref_proj)

    # Signed angle around base +Z: x_tgt_proj -> x_ref_proj.
    c = float(np.dot(x_tgt_proj, x_ref_proj))
    s = float(np.dot(z_base, np.cross(x_tgt_proj, x_ref_proj)))
    ang = float(np.arctan2(s, c))

    qz = _quat_normalize(np.array([np.cos(ang / 2.0), 0.0, 0.0, np.sin(ang / 2.0)], dtype=float))
    return _quat_normalize(_quat_mul(_quat_mul(_quat_conj(q_tgt_rel), qz), q_tgt_rel))


@dataclass(frozen=True)
class BodyXform:
    p_world: np.ndarray  # (3,)
    q_world: np.ndarray  # (4,) wxyz


def _build_body_world_xforms(root: ET.Element) -> dict[str, BodyXform]:
    """Return mapping: body_name -> world transform."""
    out: dict[str, BodyXform] = {}

    def rec(body: ET.Element, p_parent: np.ndarray, q_parent: np.ndarray) -> None:
        name = body.get("name")
        if not name:
            return
        p_local = _parse_floats(body.get("pos", "0 0 0"), 3)
        q_local = _parse_floats(body.get("quat", "1 0 0 0"), 4)
        q_world = _quat_normalize(_quat_mul(q_parent, q_local))
        p_world = p_parent + (_quat_to_rot(q_parent) @ p_local)
        out[name] = BodyXform(p_world=p_world, q_world=q_world)
        for child in body.findall("body"):
            rec(child, p_world, q_world)

    # If `root` is <worldbody>, start from each top-level body.
    if root.tag == "worldbody":
        for b in root.findall("body"):
            rec(b, np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    else:
        rec(root, np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    return out


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


def _resolve_path(p: str, base_dir: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict at top-level: {path}")
    return data


def _apply_one_target(
    *,
    reference_mjcf: Path,
    target_mjcf: Path,
    output_mjcf: Path,
    motor_bodies: list[str],
    base_body_name: str,
    align_mode: str,
    box_geom_name_template: str,
    geom_mappings: list[dict[str, str]] | None,
    reference_box_pos: np.ndarray,
    reference_rel_quat_premul: np.ndarray,
    write_geom_pos: bool,
    write_geom_quat: bool,
    float_fmt: str,
    backup_if_in_place: bool,
    in_place: bool,
) -> None:
    ref_tree = ET.parse(reference_mjcf)
    ref_root = ref_tree.getroot()
    ref_wb = _find_worldbody(ref_root)
    ref_xforms = _build_body_world_xforms(ref_wb)
    ref_bodies = _index_bodies_by_name(ref_wb)

    tgt_tree = ET.parse(target_mjcf)
    tgt_root = tgt_tree.getroot()
    tgt_wb = _find_worldbody(tgt_root)
    tgt_xforms = _build_body_world_xforms(tgt_wb)
    tgt_bodies = _index_bodies_by_name(tgt_wb)

    if base_body_name not in ref_xforms:
        raise SystemExit(f"Reference missing base body: {base_body_name}")
    if base_body_name not in tgt_xforms:
        raise SystemExit(f"Target missing base body: {base_body_name}")

    q_ref_base_w = ref_xforms[base_body_name].q_world
    q_tgt_base_w = tgt_xforms[base_body_name].q_world

    def compute_q_geom(*, ref_body: str, tgt_body: str) -> np.ndarray:
        if ref_body not in ref_xforms:
            raise KeyError(f"reference missing body {ref_body!r}")
        if tgt_body not in tgt_xforms:
            raise KeyError(f"target missing body {tgt_body!r}")
        q_ref_body_w = ref_xforms[ref_body].q_world
        q_tgt_body_w = tgt_xforms[tgt_body].q_world
        q_ref_rel = _quat_normalize(_quat_mul(_quat_conj(q_ref_base_w), q_ref_body_w))
        if reference_rel_quat_premul is not None:
            q_ref_rel = _quat_normalize(_quat_mul(reference_rel_quat_premul, q_ref_rel))
        q_tgt_rel = _quat_normalize(_quat_mul(_quat_conj(q_tgt_base_w), q_tgt_body_w))
        if align_mode == "full":
            return _quat_normalize(_quat_mul(_quat_conj(q_tgt_rel), q_ref_rel))
        if align_mode == "twist_z":
            return _twist_quat_about_body_z_to_match_ref(q_ref_rel=q_ref_rel, q_tgt_rel=q_tgt_rel)
        if align_mode == "twist_base_z":
            return _twist_quat_about_base_z_to_match_ref(q_ref_rel=q_ref_rel, q_tgt_rel=q_tgt_rel)
        raise ValueError(
            f"Unknown align_mode: {align_mode!r} (expected 'full', 'twist_z', or 'twist_base_z')"
        )

    updated = 0

    if geom_mappings:
        # Explicit mapping mode: each entry chooses which reference-body orientation should be applied
        # to which target geom.
        for m in geom_mappings:
            tgt_body = m["target_body"]
            geom_name = m["target_geom"]
            ref_body = m.get("reference_body", tgt_body)

            if tgt_body not in tgt_bodies:
                print(f"[WARN] target missing body: {tgt_body}")
                continue
            try:
                q_geom = compute_q_geom(ref_body=ref_body, tgt_body=tgt_body)
            except KeyError as e:
                print(f"[WARN] {e}")
                continue

            body_el = tgt_bodies[tgt_body]
            geom_el: ET.Element | None = None
            for g in body_el.findall("geom"):
                if g.get("name") == geom_name:
                    geom_el = g
                    break
            if geom_el is None:
                print(f"[WARN] target body {tgt_body} missing geom: {geom_name}")
                continue

            q_ref_body_w = ref_xforms[ref_body].q_world
            q_tgt_body_w = tgt_xforms[tgt_body].q_world
            p_geom = _map_vec_between_body_world_frames(q_ref_body_w, q_tgt_body_w, reference_box_pos)
            if write_geom_pos:
                geom_el.set("pos", _fmt_vec(p_geom, float_fmt))
            if write_geom_quat:
                geom_el.set("quat", _fmt_vec(q_geom, float_fmt))
            updated += 1
    else:
        # Backwards-compatible mode: one geom per motor body, using template.
        for body_name in motor_bodies:
            if body_name not in ref_xforms:
                print(f"[WARN] reference missing body: {body_name}")
                continue
            if body_name not in tgt_xforms or body_name not in tgt_bodies:
                print(f"[WARN] target missing body: {body_name}")
                continue

            q_geom = compute_q_geom(ref_body=body_name, tgt_body=body_name)

            q_ref_body_w = ref_xforms[body_name].q_world
            q_tgt_body_w = tgt_xforms[body_name].q_world
            p_geom = _map_vec_between_body_world_frames(q_ref_body_w, q_tgt_body_w, reference_box_pos)

            geom_name = box_geom_name_template.format(body=body_name)
            body_el = tgt_bodies[body_name]
            geom_el: ET.Element | None = None
            for g in body_el.findall("geom"):
                if g.get("name") == geom_name:
                    geom_el = g
                    break
            if geom_el is None:
                print(f"[WARN] target body {body_name} missing geom: {geom_name}")
                continue

            if write_geom_pos:
                geom_el.set("pos", _fmt_vec(p_geom, float_fmt))
            if write_geom_quat:
                geom_el.set("quat", _fmt_vec(q_geom, float_fmt))
            updated += 1

    if updated == 0:
        raise SystemExit("No motor box geoms updated. Check body names / geom templates.")

    # Pretty print (Python 3.9+).
    try:  # pragma: no cover
        ET.indent(tgt_tree, space="  ", level=0)
    except Exception:
        pass

    if in_place:
        if backup_if_in_place:
            backup = target_mjcf.with_suffix(target_mjcf.suffix + ".bak")
            shutil.copy2(target_mjcf, backup)
            print(f"[INFO] backup written: {backup}")
        out_path = target_mjcf
    else:
        out_path = output_mjcf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_tree.write(out_path, encoding="utf-8", xml_declaration=False)
    denom = len(geom_mappings) if geom_mappings else len(motor_bodies)
    print(f"[OK] wrote: {out_path}  (updated {updated}/{denom} entries)")


def _report_box_axes(
    *,
    mjcf_path: Path,
    motor_bodies: list[str],
    base_body_name: str,
    box_geom_name_template: str,
    geom_mappings: list[dict[str, str]] | None,
    float_fmt: str,
) -> None:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    wb = _find_worldbody(root)
    xforms = _build_body_world_xforms(wb)
    bodies = _index_bodies_by_name(wb)

    if base_body_name not in xforms:
        raise SystemExit(f"[REPORT] missing base body {base_body_name!r} in {mjcf_path}")
    R_base_w = _quat_to_rot(xforms[base_body_name].q_world)

    def fmt(v: np.ndarray) -> str:
        return _fmt_vec(np.array(v, dtype=float), float_fmt)

    print(f"\n=== REPORT box axes: {mjcf_path} ===")
    print(f"base_body_name: {base_body_name}")

    # Which geoms to report?
    # - if geom_mappings provided: report exactly those geoms (dedupe)
    # - else: report one per motor body using template (with fallbacks for unnamed boxes)
    if geom_mappings:
        items: list[tuple[str, str]] = []
        seen = set()
        for m in geom_mappings:
            key = (m["target_body"], m["target_geom"])
            if key not in seen:
                seen.add(key)
                items.append(key)
    else:
        items = [(b, box_geom_name_template.format(body=b)) for b in motor_bodies]

    for body_name, desired_geom_name in items:
        if body_name not in xforms or body_name not in bodies:
            print(f"[WARN] missing body: {body_name}")
            continue

        body_el = bodies[body_name]
        R_body_w = _quat_to_rot(xforms[body_name].q_world)

        geom_name = desired_geom_name
        geom_el: ET.Element | None = None
        for g in body_el.findall("geom"):
            if g.get("name") == geom_name:
                geom_el = g
                break
        if geom_el is None:
            # Reference `phonebot_general.xml` uses unnamed box geoms. Fall back to heuristics.
            # Prefer the green debug box (group=1) if present, otherwise take the first box geom.
            box_candidates: list[ET.Element] = []
            for g in body_el.findall("geom"):
                if g.get("type") == "box":
                    box_candidates.append(g)
            if box_candidates:
                # group=1 is used for the green debug boxes in the reference model.
                geom_el = next((g for g in box_candidates if g.get("group") == "1"), box_candidates[0])
        if geom_el is None:
            print(f"[WARN] missing geom on {body_name}: {geom_name}")
            continue

        q_geom = _parse_floats(geom_el.get("quat", "1 0 0 0"), 4)
        R_geom = _quat_to_rot(q_geom)

        # Box axes in world: body(world) * geom(local)
        R_box_w = R_body_w @ R_geom
        # Box axes in base frame: base(world)^T * box(world)
        R_box_base = R_base_w.T @ R_box_w

        x_w, y_w, z_w = R_box_w[:, 0], R_box_w[:, 1], R_box_w[:, 2]
        x_b, y_b, z_b = R_box_base[:, 0], R_box_base[:, 1], R_box_base[:, 2]

        print(f"\n[{body_name}] geom={geom_name}")
        print(f"  geom_quat(wxyz): {fmt(q_geom)}")
        print(f"  X_world: {fmt(x_w)}  Y_world: {fmt(y_w)}  Z_world: {fmt(z_w)}")
        print(f"  X_base : {fmt(x_b)}  Y_base : {fmt(y_b)}  Z_base : {fmt(z_b)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("auto_phonebot_box_placing_config.yaml")),
        help="Path to YAML config.",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Report box axes for one or more MJCF files, instead of writing outputs.",
    )
    ap.add_argument(
        "--report_mjcf",
        action="append",
        default=[],
        help="MJCF path to report (repeatable). If omitted, reports reference + config outputs.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent
    cfg = _load_yaml(cfg_path)

    reference_mjcf = _resolve_path(str(cfg["reference_mjcf"]), cfg_dir)
    motor_bodies = list(cfg["motor_bodies"])

    box_cfg = cfg.get("motor_box", {})
    base_body_name = str(box_cfg.get("base_body_name", "base_motor_link"))
    align_mode = str(box_cfg.get("align_mode", "twist_z"))
    box_geom_name_template = str(box_cfg.get("geom_name_template", "{body}_collision_box"))
    geom_mappings = box_cfg.get("geom_mappings")
    if geom_mappings is not None and not isinstance(geom_mappings, list):
        raise ValueError("motor_box.geom_mappings must be a list of dicts if provided.")
    reference_box_pos = np.array(box_cfg.get("reference_pos", [0.0, -0.012, 0.0]), dtype=float)
    reference_rel_quat_premul = np.array(box_cfg.get("reference_rel_quat_premul", [1.0, 0.0, 0.0, 0.0]), dtype=float)
    reference_rel_quat_premul = _quat_normalize(reference_rel_quat_premul)
    write_geom_pos = bool(box_cfg.get("write_pos", True))
    write_geom_quat = bool(box_cfg.get("write_quat", True))
    float_fmt = str(box_cfg.get("float_fmt", "{:.6f}"))

    run_cfg = cfg.get("run", {})
    in_place = bool(run_cfg.get("in_place", False))
    backup_if_in_place = bool(run_cfg.get("backup_if_in_place", True))

    targets = cfg.get("targets", [])
    if not isinstance(targets, list) or len(targets) == 0:
        raise ValueError("Config must include non-empty `targets` list.")

    if args.report:
        if args.report_mjcf:
            report_paths = [_resolve_path(p, cfg_dir) for p in args.report_mjcf]
        else:
            report_paths = [reference_mjcf]
            for t in targets:
                report_paths.append(_resolve_path(str(t.get("output_mjcf", t["target_mjcf"])), cfg_dir))

        for p in report_paths:
            _report_box_axes(
                mjcf_path=p,
                motor_bodies=motor_bodies,
                base_body_name=base_body_name,
                box_geom_name_template=box_geom_name_template,
                geom_mappings=geom_mappings,
                float_fmt=float_fmt,
            )
        return 0

    for t in targets:
        target_mjcf = _resolve_path(str(t["target_mjcf"]), cfg_dir)
        output_mjcf = _resolve_path(str(t.get("output_mjcf", target_mjcf)), cfg_dir)
        _apply_one_target(
            reference_mjcf=reference_mjcf,
            target_mjcf=target_mjcf,
            output_mjcf=output_mjcf,
            motor_bodies=motor_bodies,
            base_body_name=base_body_name,
            align_mode=align_mode,
            box_geom_name_template=box_geom_name_template,
            geom_mappings=geom_mappings,
            reference_box_pos=reference_box_pos,
            reference_rel_quat_premul=reference_rel_quat_premul,
            write_geom_pos=write_geom_pos,
            write_geom_quat=write_geom_quat,
            float_fmt=float_fmt,
            backup_if_in_place=backup_if_in_place,
            in_place=in_place,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

