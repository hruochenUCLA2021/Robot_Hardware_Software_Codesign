## NOTE: PhoneBot Brax policy — obs/action definition (actor vs critic)

This note describes the input/output of the trained PhoneBot joystick policy used by `CodesignEnv`.

### Which model do we deploy on the real robot?

- **Deploy only the actor** (policy network).
- The critic uses **`privileged_state`** (sim-only signals) and is **not** needed on hardware.

### Observation keys and sizes

The environment emits an observation dict with:
- **`obs["state"]`**: actor observation, shape **(52,)**
- **`obs["privileged_state"]`**: critic observation, shape **(120,)**

These keys are enforced during training/rollout by setting:
- **policy_obs_key** = `state`
- **value_obs_key** = `privileged_state`

### Actor observation: `state` (52 floats)

Built in `CodesignEnv/configs/_joystick_base_phonebot.py::_get_obs()`.

Concatenation order (with sizes):
- **noisy_gyro** (3)
- **noisy_gravity** (3)
- **command** (3): \([v_x, v_y, \omega_z]\)
- **noisy_joint_angles - default_pose** (13)
- **noisy_joint_vel** (13)
- **last_act** (13)
- **phase_feat** (4): \([cos(phase_L), cos(phase_R), sin(phase_L), sin(phase_R)]\)

Total: \(3 + 3 + 3 + 13 + 13 + 13 + 4 = 52\).

Important details:
- The actor **does not** see contact or base linear velocity (NoLinearVel style).
- The IMU frame is taken from the configured `imu` site; gravity is computed as:
  `data.site_xmat[imu_site].T @ [0, 0, -1]`.

### Critic observation: `privileged_state` (120 floats)

Prefix: **state_with_linvel** (55 floats):
- **noisy_linvel** (3)
- then the same signals as actor `state` (gyro, gravity, command, joints, last_act, phase_feat)

Then appended signals:
- **gyro** (3)
- **accelerometer** (3)
- **gravity** (3)
- **linvel** (3) (IMU-local)
- **global_angvel** (3)
- **joint_angles - default_pose** (13)
- **joint_vel** (13)
- **root_height** (1)
- **actuator_force** (13)
- **contact** (2) (feet vs floor)
- **feet_vel** (6) (left/right foot global linear velocity, xyz each)
- **feet_air_time** (2)

Total: \(55 + 41 + 24 = 120\).

### Action: actor output (13 floats)

The policy outputs a vector `action` with shape **(13,)**.

In the environment, this is mapped to motor targets as:
- **motor_targets = clip(default_pose + action * action_scale, soft_lower, soft_upper)**
- PhoneBot uses **`action_scale = 1.0`**.

So **action is a normalized delta around `default_pose`** (in radians), then clipped.

### Default pose (PhoneBot) used by the environment

From the env instance (example checkpoint `PhonebotJoystickFlatTerrain_flat_ok_v6/final`):

`default_pose` (13):
- `[-0.44, 0.0, 0.0, 0.785, -0.377, 0.0, 0.44, 0.0, 0.0, -0.785, 0.377, 0.0, 0.0]`

### Action ordering (matches actuator ordering in MJCF)

In `models/model_phonebot/phonebot_general.xml` the `<actuator>` block lists:

1. `l_hip_pitch`
2. `l_hip_roll`
3. `l_hip_thigh`
4. `l_hip_calf`
5. `l_ankle_pitch`
6. `l_ankle_roll`
7. `r_hip_pitch`
8. `r_hip_roll`
9. `r_hip_thigh`
10. `r_hip_calf`
11. `r_ankle_pitch`
12. `r_ankle_roll`
13. `base_to_trunk`

This is the order of:
- `action[i]`
- `data.ctrl[i]`
- `mjx_model.nu` indexing

### Deployment checklist (Android / LiteRT)

- Export only the actor policy to `.tflite`.
- Feed **exactly 52 float32 values** for `state` (or the checkpoint’s expected dim).
- Interpret output as **13 float32 actions**; map to targets via `default_pose + action`.

