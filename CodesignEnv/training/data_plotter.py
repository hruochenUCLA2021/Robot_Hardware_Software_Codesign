#!/usr/bin/env python3
"""Plot qpos and qvel from rollout JSON records.

Usage:
    python data_plotter.py                         # uses default config
    python data_plotter.py /path/to/config.yaml    # custom config
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_record(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_signals(
    data: np.ndarray,
    labels: list[str],
    title_prefix: str,
    dt: float,
    out_path: str,
    figsize_per: tuple[float, float] = (12, 2.5),
    dpi: int = 150,
):
    """Create one tall figure with a subplot per signal column."""
    n_steps, n_cols = data.shape
    time = np.arange(n_steps) * dt

    fig, axes = plt.subplots(
        n_cols, 1,
        figsize=(figsize_per[0], figsize_per[1] * n_cols),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(title_prefix, fontsize=14, y=1.0)

    for i in range(n_cols):
        ax = axes[i, 0]
        label = labels[i] if i < len(labels) else f"dim_{i}"
        ax.plot(time, data[:, i], linewidth=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def process_record(json_path: str, plot_cfg: dict):
    """Load one JSON record and produce qpos + qvel plot images."""
    record = _load_record(json_path)
    rollout_name = record.get("rollout_name", "unknown")
    dt = float(record.get("dt", 0.002))
    qpos = np.array(record["qpos"], dtype=np.float64)
    qvel = np.array(record["qvel"], dtype=np.float64)
    qpos_labels = record.get("qpos_labels", [f"qpos_{i}" for i in range(qpos.shape[1])])
    qvel_labels = record.get("qvel_labels", [f"qvel_{i}" for i in range(qvel.shape[1])])

    json_dir = os.path.dirname(os.path.abspath(json_path))
    plot_dir = os.path.join(json_dir, f"plots_{rollout_name}")
    os.makedirs(plot_dir, exist_ok=True)

    figsize_per = tuple(plot_cfg.get("figsize_per_subplot", [12, 2.5]))
    dpi = int(plot_cfg.get("dpi", 150))

    print(f"[{rollout_name}] {qpos.shape[0]} steps, "
          f"qpos dim={qpos.shape[1]}, qvel dim={qvel.shape[1]}")

    _plot_signals(
        qpos, qpos_labels,
        title_prefix=f"{rollout_name} — qpos",
        dt=dt,
        out_path=os.path.join(plot_dir, "qpos.png"),
        figsize_per=figsize_per,
        dpi=dpi,
    )
    _plot_signals(
        qvel, qvel_labels,
        title_prefix=f"{rollout_name} — qvel",
        dt=dt,
        out_path=os.path.join(plot_dir, "qvel.png"),
        figsize_per=figsize_per,
        dpi=dpi,
    )


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _THIS_DIR, "data_plotter_config.yaml"
    )
    print(f"Using plotter config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    records = cfg.get("records", [])
    plot_cfg = cfg.get("plot", {})

    if not records:
        print("No records listed in config; nothing to plot.")
        return

    for entry in records:
        json_path = entry if isinstance(entry, str) else entry.get("path", "")
        if not os.path.isabs(json_path):
            json_path = os.path.join(_THIS_DIR, json_path)
        if not os.path.isfile(json_path):
            print(f"[SKIP] File not found: {json_path}")
            continue
        process_record(json_path, plot_cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
