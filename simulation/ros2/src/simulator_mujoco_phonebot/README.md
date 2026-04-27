## simulator_mujoco_phonebot

Real-time **MuJoCo CPU** simulation + viewer for Phonebot, driven by a **trained
CodesignEnv joystick policy** and controlled by the keyboard.

### Controls

- Arrow keys: command \(v_x, v_y\)
- `a` / `d`: command yaw rate \(w_z\)
- Hold **Shift**: fast (full range)
- Hold **Ctrl**: slow (quarter range)
- No modifier: normal (half range)
- Space: stop (zero command)
- Esc: quit

### Run

After building your ROS2 workspace with `colcon build` and sourcing it:

```bash
ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard \
  --ros-args \
  -p checkpoint_dir:=/ABS/PATH/TO/checkpoints/PhonebotJoystickFlatTerrainAlterFV2TorqueAwared_flat/final \
  -p home_keyframe_name:=home_straight
```

### Useful parameters

- `env_name` (str): registry env name
- `task` (str): task name, e.g. `flat_terrain_alternative_imu_fv2_torque`
- `checkpoint_dir` (str): path to the Brax checkpoint folder (**required**)
- `home_keyframe_name` (str): keyframe name in the scene XML
- `render` (bool): open interactive viewer
- `control_hz` (float): default 50 Hz
- `sim_hz` (float): default 500 Hz (10 substeps at 50 Hz)
- `max_vx`, `max_vy`, `max_wz` (float): full-speed command limits

