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
## Torque-control realtime (torque-aware policies)
ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/ABS/PATH/TO/checkpoints/PhonebotJoystickFlatTerrainAlterFV2TorqueAwared_flat/final \
  -p home_keyframe_name:=home_straight


ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/checkpoints/PhonebotJoystickFlatTerrainAlterFV2TorqueAwared_flat_curriculum_v1_home_straight_v1/final \
  -p policy_format:=brax \
  -p home_keyframe_name:=home_straight


ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_home_straight_v1_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p render:=true \
  -p home_keyframe_name:=home_straight

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_home_straight_v1_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p show_contact:=true \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_full_collision.xml

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_rough_alter_fv2_torque_awared_home_straight_v1_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_full_collision.xml

pip install ai-edge-litert

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_home_straight_v1_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight


ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_ankle_collision_home_straight_v1_look_unstable_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_full_collision.xml

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_half_v2_89128960_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_full_collision.xml

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_half_v2_178257920_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_full_collision.xml

ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_half_v2_178257920_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_ankle_collision.xml

  
ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_ankle_collision_home_straight_v3_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_ankle_collision.xml


ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_ankle_collision_home_straight_v4_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_ankle_collision.xml



ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_ankle_collision_home_straight_v6_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_ankle_collision.xml



ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_torque \
  --ros-args \
  -p checkpoint_dir:=/media/hrc/T7_UBUNTU_ONLY/Codesign_ChopstickBot_all_files/Robot_Hardware_Software_Codesign/CodesignEnv/training/exported_tflite/phonebot_flat_alter_fv2_torque_awared_ankle_collision_home_straight_v5_new_reward_for_deadzone_behavior_actor.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert \
  -p home_keyframe_name:=home_straight \
  -p xml_path:=models/model_phonebot_fred_v2_torque_version/scene_joystick_flat_terrain_alter_v2_ankle_collision.xml


```

## Position-control realtime (position-controlled policies)

Example (Phonebot FV2 position-control policy):

```bash
ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_position \
  --ros-args \
  -p env_name:=PhonebotJoystickFlatTerrainAlterFV2 \
  -p task:=flat_terrain_alternative_imu_fv2 \
  -p checkpoint_dir:=/ABS/PATH/TO/checkpoints/PhonebotJoystickFlatTerrainAlterFV2_flat/final \
  -p policy_format:=brax \
  -p home_keyframe_name:=home_straight
```

Example (Chopstickbot position-control joystick policy on a leglen sweep XML):

```bash
ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard_position \
  --ros-args \
  -p env_name:=ChopstickbotJoystickFlatTerrainAlter \
  -p task:=flat_terrain_alternative_imu \
  -p checkpoint_dir:=/ABS/PATH/TO/checkpoints/ChopstickbotJoystickFlatTerrainAlter_flat/final \
  -p policy_format:=brax \
  -p xml_path:=models/model_chopstickbot_leglen_sweep/len_0.20m/scene_joystick_flat_terrain_chopstickbot.xml
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

### Cross-repo realtime simulators (for comparison)

These nodes run **MuJoCo CPU** simulation + viewer but load policies from
different repos/formats so you can compare behavior.

```bash
## 1) BoosterGym TorchScript policy (T1, serial model with arms+waist)
ros2 run simulator_mujoco_phonebot realtime_simulation_booster_gym_policy \
  --ros-args \
  -p config_path:=reference/booster_gym/envs/T1.yaml \
  -p xml_path:=reference/booster_gym/resources/T1/T1_locomotion.xml \
  -p policy_path:=checkpoints_to_test/booster_gym_t1/logs/2026-05-12-23-31-29/nn/model_10000.pt

## 2) HERMES Hi Brax PPO policy (joystick / NoLinearVel)
ros2 run simulator_mujoco_phonebot realtime_simulation_hermes_hi_policy \
  --ros-args \
  -p env_name:=HiJoystickFlatTerrainNoLinearVel \
  -p checkpoint_dir:=checkpoints_to_test/HERMES_hi/HiJoystickFlatTerrainNoLinearVel_flat/final \
  -p xml_path:=reference/HERMES_simulation_model/model_modification/mjcf/scene_joystick_flat_terrain.xml

## 2b) HERMES Hi Brax PPO policy (joystick WITH velocity input)
ros2 run simulator_mujoco_phonebot realtime_simulation_hermes_hi_policy \
  --ros-args \
  -p env_name:=HiJoystickFlatTerrain \
  -p checkpoint_dir:=checkpoints_to_test/HERMES_hi/HiJoystickFlatTerrain_v6_good/final \
  -p xml_path:=reference/HERMES_simulation_model/model_modification/mjcf/scene_joystick_flat_terrain.xml

## 3) MuJoCo-Playground T1 Brax PPO policy (flat terrain)
ros2 run simulator_mujoco_phonebot realtime_simulation_playground_t1_policy \
  --ros-args \
  -p env_name:=T1JoystickFlatTerrain \
  -p checkpoint_dir:=checkpoints_to_test/mujoco_playground/T1JoystickFlatTerrain/final \
  -p xml_path:=reference/mujoco_playground/_src/locomotion/t1/xmls/scene_mjx_feetonly_flat_terrain.xml
```



```bash
## 1) BoosterGym TorchScript policy (T1, serial model with arms+waist)
ros2 run simulator_mujoco_phonebot realtime_simulation_booster_gym_policy \
  --ros-args \
  -p config_path:=reference/booster_gym/deploy/configs/T1.yaml \
  -p xml_path:=reference/booster_gym/resources/T1/T1_serial.xml \
  -p policy_path:=/ABS/PATH/TO/T1.pt

## 2) HERMES Hi Brax PPO policy (joystick / NoLinearVel)
ros2 run simulator_mujoco_phonebot realtime_simulation_hermes_hi_policy \
  --ros-args \
  -p env_name:=HiJoystickFlatTerrainNoLinearVel \
  -p checkpoint_dir:=/ABS/PATH/TO/checkpoints/HiJoystickFlatTerrainNoLinearVel_flat/final \
  -p xml_path:=reference/HERMES_simulation_model/model_modification/mjcf/scene_joystick_flat_terrain.xml

## 2b) HERMES Hi Brax PPO policy (joystick WITH velocity input)
ros2 run simulator_mujoco_phonebot realtime_simulation_hermes_hi_policy \
  --ros-args \
  -p env_name:=HiJoystickFlatTerrain \
  -p checkpoint_dir:=/ABS/PATH/TO/checkpoints/HiJoystickFlatTerrain_flat/final \
  -p xml_path:=reference/HERMES_simulation_model/model_modification/mjcf/scene_joystick_flat_terrain.xml

## 3) MuJoCo-Playground T1 Brax PPO policy (flat terrain)
ros2 run simulator_mujoco_phonebot realtime_simulation_playground_t1_policy \
  --ros-args \
  -p env_name:=T1JoystickFlatTerrain \
  -p checkpoint_dir:=/ABS/PATH/TO/checkpoints/T1JoystickFlatTerrain/final \
  -p xml_path:=reference/mujoco_playground/_src/locomotion/t1/xmls/scene_mjx_feetonly_flat_terrain.xml
```