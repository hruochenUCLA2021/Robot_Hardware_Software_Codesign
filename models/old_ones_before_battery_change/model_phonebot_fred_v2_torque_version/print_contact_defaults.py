#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _id2name(model, objtype: int, objid: int) -> str:
    import mujoco

    nm = mujoco.mj_id2name(model, objtype, objid)
    return nm if nm else f"{objtype}:{objid}"


def _summarize_unique_rows(*, title: str, values, names: list[str], max_examples: int = 5) -> None:
    import numpy as np

    v = np.asarray(values)
    if v.ndim == 1:
        v = v.reshape(-1, 1)

    uniq, inv, counts = np.unique(v, axis=0, return_inverse=True, return_counts=True)
    print(f"\n== {title} ==")
    for i, row in enumerate(uniq):
        idxs = np.nonzero(inv == i)[0]
        ex = [names[int(k)] for k in idxs[:max_examples]]
        row_str = " ".join(f"{float(x):.6g}" for x in row.tolist())
        print(f"- value: [{row_str}]  count={int(counts[i])}  examples={ex}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Print compiled MuJoCo contact defaults used by a model.")
    ap.add_argument(
        "--xml",
        default=str(
            Path(__file__).resolve().parents[1] / "model_phonebot" / "phonebot_general.xml"
        ),
        help="MJCF model path (default: model_phonebot/phonebot_general.xml).",
    )
    args = ap.parse_args()

    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {xml_path}")

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(f"MuJoCo python version: {mujoco.__version__}")
    print(f"Loaded: {xml_path}")
    print(f"ngeom={model.ngeom}  npair={model.npair}  nconmax={model.nconmax}")

    # Global contact override (usually zeros => disabled).
    print("\n== option contact override (opt.o_*) ==")
    print(f"opt.o_solref:   {model.opt.o_solref[0]:.6g} {model.opt.o_solref[1]:.6g}")
    print(
        "opt.o_solimp:   "
        + " ".join(f"{float(x):.6g}" for x in model.opt.o_solimp[:].tolist())
    )
    print(
        "opt.o_friction: "
        + " ".join(f"{float(x):.6g}" for x in model.opt.o_friction[:].tolist())
    )

    geom_names = [_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]

    # All geoms (including non-colliding ones).
    _summarize_unique_rows(title="geom_solref (all geoms)", values=model.geom_solref, names=geom_names)
    _summarize_unique_rows(title="geom_solimp (all geoms)", values=model.geom_solimp, names=geom_names)
    _summarize_unique_rows(title="geom_friction (all geoms)", values=model.geom_friction, names=geom_names)
    _summarize_unique_rows(title="geom_condim (all geoms)", values=np.asarray(model.geom_condim), names=geom_names)
    _summarize_unique_rows(title="geom_priority (all geoms)", values=np.asarray(model.geom_priority), names=geom_names)

    # Collidable subset: contype!=0 and conaffinity!=0 are a simple proxy for "intended to collide".
    contype = np.asarray(model.geom_contype)
    conaff = np.asarray(model.geom_conaffinity)
    coll_idx = np.nonzero((contype != 0) & (conaff != 0))[0]
    coll_names = [geom_names[int(i)] for i in coll_idx.tolist()]
    print(f"\nCollidable geoms (contype!=0 and conaffinity!=0): {len(coll_idx)}/{model.ngeom}")
    if len(coll_idx):
        _summarize_unique_rows(
            title="geom_solref (collidable only)", values=model.geom_solref[coll_idx], names=coll_names
        )
        _summarize_unique_rows(
            title="geom_solimp (collidable only)", values=model.geom_solimp[coll_idx], names=coll_names
        )
        _summarize_unique_rows(
            title="geom_friction (collidable only)", values=model.geom_friction[coll_idx], names=coll_names
        )
        _summarize_unique_rows(
            title="geom_condim (collidable only)", values=np.asarray(model.geom_condim)[coll_idx], names=coll_names
        )
        _summarize_unique_rows(
            title="geom_priority (collidable only)", values=np.asarray(model.geom_priority)[coll_idx], names=coll_names
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

