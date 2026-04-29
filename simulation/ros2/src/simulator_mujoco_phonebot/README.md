## simulator_mujoco_phonebot

Real-time **MuJoCo CPU** simulation + viewer for Phonebot, driven by a **trained
CodesignEnv joystick policy** and controlled by the keyboard.

### Controls

- Arrow keys: command \(v_x, v_y\)
- `1` / `3`: command yaw rate \(w_z\)
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


ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/checkpoints/PhonebotJoystickFlatTerrainAlterFV2TorqueAwared_flat_curriculum_v1_home_straight_v1/final \
  -p policy_format:=brax \
  -p home_keyframe_name:=home_straight


ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_home_straight_v1_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p render:=true \
  -p home_keyframe_name:=home_straight

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_home_straight_v1_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p show_contact:=true \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_full_collision.xml

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_rough_alter_fv2_torque_awared_home_straight_v1_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_full_collision.xml

pip install ai-edge-litert

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_home_straight_v1_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight
```

If the TFLite mode hangs when importing TensorFlow, install LiteRT (recommended, supports Python 3.12):

```bash
pip install ai-edge-litert
```

If you see `ModuleNotFoundError: No module named 'CodesignEnv'`, make sure you
run from inside the repo (so the node can find `Robot_Hardware_Software_Codesign/`
via `cwd`), for example:

```bash
cd /ABS/PATH/TO/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/simulation/ros2
```

### Useful parameters

- `env_name` (str): registry env name
- `task` (str): task name, e.g. `flat_terrain_alternative_imu_fv2_torque`
- `xml_path` (str): optional scene XML override (absolute or relative to `Robot_Hardware_Software_Codesign/`).
- `checkpoint_dir` (str): path to a **Brax checkpoint folder** or a **.tflite** file (**required**)
- `policy_format` (str): `auto|brax|tflite` (default `auto`)
- `tflite_num_threads` (int): TF Lite CPU threads (default 1)
- `tflite_backend` (str): `auto|litert|tflite_runtime|tensorflow` (default `auto`)
- `home_keyframe_name` (str): keyframe name in the scene XML
- `render` (bool): open interactive viewer
- `show_contact` (bool): show contact points/forces in the viewer
- `control_hz` (float): default 50 Hz
- `sim_hz` (float): default 500 Hz (10 substeps at 50 Hz)
- `max_vx`, `max_vy`, `max_wz` (float): full-speed command limits

