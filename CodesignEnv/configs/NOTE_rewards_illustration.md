## Rewards/config illustration (Phonebot torque-aware ankle-collision joystick)

This note explains the `default_config()` fields in:

- `Robot_Hardware_Software_Codesign/CodesignEnv/configs/_joystick_base_phonebot_torque_awared_ankle_collision.py` (see lines ~27–160)

It is meant as a “what does this knob do?” reference for tuning.

---

## Global sim/control timing

- **`ctrl_dt`** (seconds): controller period. Policy action is applied once per `ctrl_dt`.
  - Example: `0.02` → 50 Hz control.
- **`sim_dt`** (seconds): MuJoCo integration timestep.
  - Example: `0.002` → 500 Hz sim.
- **`episode_length`** (steps): maximum steps per episode (in *control* steps).
- **`action_repeat`**: how many times to repeat each action. (Often `1` for joystick tasks.)
- **`action_scale`**: scales the policy action before being used as a target (or before mapping into the motor controller).

---

## Initialization / keyframes

- **`home_keyframe_name`**: which MJCF keyframe to use for initial `qpos` (and often for defining the “default pose”).

---

## Torque controller + per-episode randomization

- **`motor_params_path`**: YAML file with motor/controller defaults (gains, torque envelope parameters, etc.), loaded at env init.

### `actuator_noise_config` (ToddlerBot-style multiplicative per-episode noise)

These are episode-level random scalings. If enabled, the env samples a factor for each parameter in the given range and multiplies the nominal parameter.

- **`enable`**: master switch.
- **`kp_range`, `kd_range`**: scale PD gains.
- **`tau_max_range`**: scale torque limit (peak).
- **`q_dot_tau_max_range`, `q_dot_max_range`**: scale velocity-dependent torque/velocity envelope terms (if used by your motor model).
- **`kd_min_range`**: scale minimum damping term (if your controller clamps `kd`).
- **`tau_brake_max_range`, `tau_q_dot_max_range`**: scale braking / backdrive envelope terms (if used).
- **`passive_active_ratio_range`**: scale passive-vs-active blend ratio (if used by motor model).

Practical intent: make the policy robust to motor/model mismatch.

---

## Observation noise

- **`tracking_sigma`**: affects how sharply command-tracking rewards drop with error. Smaller → harsher tracking error penalty; larger → more forgiving.

### `noise_config`

- **`level`**: global multiplier for observation noise.
- **`scales`**: per-signal noise magnitudes (units match each signal).
  - **`joint_pos`**: joint angle noise (rad).
  - **`joint_vel`**: joint velocity noise (rad/s).
  - **`gravity`**: gravity vector noise (unitless).
  - **`linvel`**: base linear velocity noise (m/s).
  - **`gyro`**: angular velocity noise (rad/s).

---

## Reward config

### Sign convention

In this env, the reward dictionary contains a mix of:

- “good” terms (e.g. tracking) that are positive by construction (often \( \exp(-\mathrm{error}) \))
- “cost” terms (e.g. torques) that are positive magnitudes (abs/square/etc.)

The final scalar reward is typically a weighted sum:

> **`total = Σ (scale[k] * term[k])`**

So:

- **positive `scale`** → encourages increasing the term
- **negative `scale`** → penalizes the term

### `reward_config.scales` items (what each term is “for”)

#### Command tracking (the “main task”)

- **`tracking_lin_vel`**: track commanded XY base velocity (higher is better).
- **`tracking_ang_vel`**: track commanded yaw rate (higher is better).

#### Stability / body motion regularizers

- **`lin_vel_z`** (penalty): vertical base velocity (jumping/bouncing).
- **`ang_vel_xy`** (penalty): roll/pitch angular velocity magnitude.
- **`orientation`** (penalty): tilt away from upright (uses gravity vector XY components).

#### Energy / smoothness regularizers

These are typically “costs” computed as magnitudes; use **negative** scales to penalize.

- **`torques`**: \(\sum | \tau |\) (L1 torque usage).
- **`torques_square`**: mean(\(\tau^2\)) (L2 torque usage; punishes spikes more).
- **`energy`**: \(\sum | \dot{q} \cdot \tau |\) (mechanical power magnitude).
- **`energy_square`**: mean((\(\dot{q}\cdot\tau)^2\)).
- **`action_rate`**: \(\sum (a_t - a_{t-1})^2\) (smooth actions).
- **`dof_acc`**: \(\sum \ddot{q}^2\) (smooth accelerations).
- **`dof_vel`**: \(\sum \dot{q}^2\) (discourage thrashing).

#### Foot/leg behavior shaping

- **`feet_air_time`** (reward): encourages stepping rhythm by rewarding “time in air” until first contact, gated by non-zero command.
- **`feet_clearance`** (penalty): encourages swing-foot clearance during motion.
  - Penalizes deviation from target height **while the foot is moving in XY** (weighted by horizontal foot speed).
  - This is *not* about separating feet; it’s about avoiding toe scuff during swing.
- **`feet_height`** (penalty): swing *peak* height accuracy (applied on first contact).
  - Uses `swing_peak / max_foot_height` error squared.
  - Complementary to `feet_clearance`: one is “during swing”, one is “peak at touchdown”.
- **`feet_distance`** (penalty): discourages a stance that is too narrow.
  - It penalizes when the lateral distance between feet is **less than a fixed threshold** (hard-coded as `0.2 m` in the cost function).
  - If your feet look too close, this is the **main knob** to increase (make the scale more negative in magnitude).
- **`foot_collision`** (penalty): penalizes direct foot ↔ foot contact (binary).
- **`body_collision`** (penalty): penalizes foot ↔ (ankle/knee/hip-roll proxy) contacts (binary).

#### Pose regularizers (keep a “reasonable” joint posture)

- **`joint_deviation_knee`** (penalty): keep knees near default pose.
- **`joint_deviation_hip`** (penalty): keep hips near default pose (often gated by lateral command).
- **`dof_pos_limits`** (penalty): penalize joint angles outside soft limits.
- **`pose`** (penalty): weighted squared deviation from default pose (global posture).

#### Survival shaping

- **`alive`** (reward): constant reward while alive (helps prevent early termination solutions).

### `reward_config.max_foot_height`

- **`max_foot_height`** (meters): the target swing height used by both:
  - `feet_clearance` (instantaneous swing tracking)
  - `feet_height` (swing peak error at touchdown)

If you increase it, you’re asking the gait to lift feet higher (more clearance, but often more energy and possibly more instability).

---

## Push/random disturbances

### `push_config`

- **`enable`**: whether random pushes are applied.
- **`interval_range`** (seconds): time between pushes.
- **`magnitude_range`**: push strength range (implementation-dependent units; often impulse/force-like).

Intent: robustness to disturbances.

---

## Command ranges (joystick task distribution)

These define the sampling ranges for commands used during training/rollouts:

- **`lin_vel_x`** (m/s): forward/back command range.
- **`lin_vel_y`** (m/s): lateral command range.
- **`ang_vel_yaw`** (rad/s): yaw rate command range.

Wider ranges usually produce more diverse behaviors but can make learning harder.

---

## Quick “my feet are too close” tuning checklist

- **First knob**: make `scales.feet_distance` more negative (stronger penalty for narrow stance).
- **Second knob**: increase `scales.foot_collision` (if they actually touch).
- If changing stance makes the gait unstable, you may need to retune pose penalties (`pose`, hip/knee deviation) to allow a wider gait.

