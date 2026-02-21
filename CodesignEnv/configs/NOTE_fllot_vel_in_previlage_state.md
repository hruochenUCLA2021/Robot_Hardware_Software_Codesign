# NOTE: foot velocity in `privileged_state` (world frame)

This repo’s joystick env (`CodesignEnv/configs/_joystick_common.py`) includes per-foot velocity in the **critic-only** observation (`privileged_state`) using MuJoCo sensors:

- `left_foot_global_linvel`
- `right_foot_global_linvel`

These are **world/global-frame** linear velocities of the foot sites.

## Why world-frame foot velocity?

- **Slip is defined relative to the ground**: the ground is stationary in the **world** frame, so a “foot sliding on the ground” measure is naturally expressed in world \(x,y\) (or world horizontal plane).
- **Matches the HERMES/T1 design**: the reference HERMES `hi_joystick_NoLinearVel.py` packs `*_global_linvel` into `privileged_state` and uses it for slip-related terms (e.g., “feet slip vs ground”).
- **Foot-local frame is not stable**: the foot/link frame rotates with the leg, so “local x/y” changes meaning over time. That makes a slip measure less direct.

## Why is this in `privileged_state` (critic) and not actor `state`?

These signals are **simulation-only** / hard to estimate robustly on hardware. We keep them for the critic to improve value learning while the actor sees only hardware-plausible signals.

## If you ever want foot velocity in IMU-local coordinates

You can convert world foot velocity \(v_w\) to IMU-local \(v_{imu}\) using the IMU site rotation:

\[
v_{imu} = R_{imu}^T \, v_w
\]

Where \(R_{imu}\) is the \(3\times 3\) rotation matrix from the IMU site (MuJoCo: `data.site_xmat[imu_site_id]`).

