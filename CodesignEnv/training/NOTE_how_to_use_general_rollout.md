## General Phonebot rollout (MuJoCo CPU) — how to use

This note documents how to use:

- `CodesignEnv/training/load_and_rollout_phonebot_joystick_general.py`
- `CodesignEnv/training/rollout_phonebot_joystick_general_config.yaml`

It supports:

- **MuJoCo CPU** simulation (offline) and MP4 video recording
- **Brax checkpoint** policy loading (fast path `load_policy` → fallback restore)
- **TFLite** policy loading (LiteRT recommended)
- Optional **JSON recording** of `qpos/qvel/qacc/actuator_force`
- Optional **matplotlib plots** generated from that JSON

---

## Run it

From `Robot_Hardware_Software_Codesign/CodesignEnv/training/`:

```bash
python ./load_and_rollout_phonebot_joystick_general.py
python ./load_and_rollout_phonebot_joystick_general.py ./rollout_phonebot_joystick_general_config.yaml
```

---

## Config structure (jobs)

The YAML is a set of named jobs:

- `job_name`: which job(s) to run
  - string: run one job
  - list of strings: run multiple jobs
  - `"all"`: run all jobs in `jobs:`
- `jobs:`: map from job name → job config

Each job contains:

- `policy_format`: `brax` | `tflite` | `auto`
- `policy_path`: checkpoint dir OR `.tflite` path (depending on format)
- `xml_path`: scene MJCF to simulate in MuJoCo CPU
- `env_name` + `task` (+ optional `env_config_path`): used for Brax fallback restore and torque-controller reuse (details below)
- rollout/render params: `physics_hz`, `control_hz`, `fps`, `render_every`, `show_contact`
- output params: `output_dir`
- `rollouts`: list of individual rollouts (command/camera/max_steps/etc.)

---

## Paths (absolute vs relative)

### `xml_path`

- **Absolute** path works directly.
- **Relative** path is resolved relative to `Robot_Hardware_Software_Codesign/` (project root).

So prefer:

- `models/model_phonebot_fred_v2_torque_version/scene_...xml`

(not `Robot_Hardware_Software_Codesign/models/...`, because that double-prefixes).

### `policy_path`

- **Absolute** path works directly.
- **Relative** path is resolved relative to `CodesignEnv/training/` (the script directory).

### `output_dir`

- **Relative** `output_dir` is resolved relative to `CodesignEnv/training/`.
- The directory does **not** need to exist; it will be created.

---

## XML vs env/task priority (important)

### Which one decides the MuJoCo scene?

**`xml_path` always decides the MuJoCo CPU model.**

The script will load exactly that XML in MuJoCo (`mujoco.MjModel.from_xml_path(xml_path)`).

### Then why keep `env_name` / `task`?

`env_name` + `task` are still useful for:

- **Brax restore fallback**: if a Brax checkpoint does not have a usable `ppo_network_config.json`, the script falls back to restoring via `ppo.train(... restore_checkpoint_path=...)`, which needs an MJX env instance.
- **Torque controller reuse**: if the MJX env has `env._motor_controller`, the script will use that controller to convert action targets → torque for MuJoCo CPU.
- **Action scaling**: if `action_scale: null`, the script will try to read `env._config.action_scale`.

---

## Policy loading

### Brax checkpoint (`policy_format: brax`)

`policy_path` can be:

- the leaf checkpoint dir itself
- a parent containing `final/`
- a parent containing numeric step subdirs (the script picks the latest)

Load order:

1. Try `ppo_checkpoint.load_policy(ckpt_leaf)` (**fast**) if `ppo_network_config.json` exists and is compatible.
2. Fallback to `ppo.train(... restore_checkpoint_path=ckpt_leaf)` (**slower**, requires `env_name` + `task`).

### TFLite (`policy_format: tflite`)

Keys:

- `tflite_backend`: `litert` (recommended) or `tensorflow`
- `tflite_num_threads`: interpreter thread count

---

## Rollouts list

Each entry in `rollouts:` typically contains:

- `name`: used for log/JSON naming
- `out`: output MP4 filename (relative to `output_dir`)
- `command`: `[vx, vy, yaw_rate]` (same convention as joystick env)
- `start_pose`: `[x, y, yaw]` (optional; defaults to job’s `start_pose_default`)
- `camera`: usually `track` or `null` (null means “default camera/view”)
- `max_steps`, `render_every`: optional overrides

---

## Recording JSON + auto plotting (new)

The general rollout can save a JSON record compatible with:

- `CodesignEnv/training/data_plotter.py`

### Defaults (recommended)

At the **top level** of the YAML:

- `record_json: true`
- `auto_plot: true`
- optional `plot:` section (controls figure sizes / dpi)

### Disable for a whole job

Inside a job:

```yaml
record_json: false
auto_plot: false
```

### Override per rollout

Inside a rollout item:

- Disable:

```yaml
record_json: false
```

- Custom JSON filename/path (relative → `output_dir/`):

```yaml
record_json: my_run_record.json
```

### What gets recorded

The JSON contains:

- `qpos`, `qvel`, `qacc`
- `actuator_force` (used as “torque proxy”)
- label arrays (`qpos_labels`, `qvel_labels`, `qacc_labels`, `actuator_force_labels`)
- metadata (`xml_path`, `home_keyframe_name`, `command`, `dt`, etc.)

### Where plots go

For each JSON record `<name>_record.json`, plots are written to:

- `output_dir/plots_<name>_record/`
  - `qpos.png`
  - `qvel.png`
  - `qacc.png` (if available)
  - `torque.png` (if available)

---

## Common pitfalls

- **`xml_path not found` with duplicated prefix**:
  - If you see `.../Robot_Hardware_Software_Codesign/Robot_Hardware_Software_Codesign/...`,
    remove the leading `Robot_Hardware_Software_Codesign/` from `xml_path` in YAML.
- **Camera name errors**:
  - If the named camera doesn’t exist, set `camera: null`.
- **Brax fallback restore needs env**:
  - If you use `policy_format: brax` and the checkpoint can’t be loaded via `load_policy`,
    you must provide `env_name` + `task` so the script can build the MJX env for restore.

