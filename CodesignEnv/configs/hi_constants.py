# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for CodesignEnv robots (PhoneBot + ChopstickBot)."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = epath.Path(__file__).parent.parent  # .../Robot_Hardware_Software_Codesign/CodesignEnv
PROJECT_ROOT = ROOT_PATH.parent  # .../Robot_Hardware_Software_Codesign
MODELS_ROOT = PROJECT_ROOT / "models"

PHONEBOT_XML_ROOT = MODELS_ROOT / "model_phonebot"
CHOPSTICKBOT_XML_ROOT = MODELS_ROOT / "model_chopstickbot"
PHONEBOT_FV2_XML_ROOT = MODELS_ROOT / "model_phonebot_fred_v2"
PHONEBOT_FV2_TORQUE_XML_ROOT = MODELS_ROOT / "model_phonebot_fred_v2_torque_version"
CHOPSTICKBOT_TORQUE_XML_ROOT = MODELS_ROOT / "model_chopstickbot_torque_version"

PHONEBOT_JOYSTICK_FLAT_TERRAIN_XML = PHONEBOT_XML_ROOT / "scene_joystick_flat_terrain.xml"
PHONEBOT_JOYSTICK_ROUGH_TERRAIN_XML = PHONEBOT_XML_ROOT / "scene_joystick_rough_terrain.xml"
PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_XML = (
    PHONEBOT_XML_ROOT / "scene_joystick_flat_terrain_alternative_imu.xml"
)
PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_XML = (
    PHONEBOT_XML_ROOT / "scene_joystick_rough_terrain_alternative_imu.xml"
)
PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_XML = (
    PHONEBOT_FV2_XML_ROOT / "scene_joystick_flat_terrain_alter_v2.xml"
)
PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_XML = (
    PHONEBOT_FV2_XML_ROOT / "scene_joystick_rough_terrain_alter_v2.xml"
)

PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_XML = (
    PHONEBOT_FV2_TORQUE_XML_ROOT / "scene_joystick_flat_terrain_alter_v2.xml"
)
PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_XML = (
    PHONEBOT_FV2_TORQUE_XML_ROOT / "scene_joystick_rough_terrain_alter_v2.xml"
)

PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_ANKLE_COLLISION_XML = (
    PHONEBOT_FV2_TORQUE_XML_ROOT / "scene_joystick_flat_terrain_alter_v2_ankle_collision.xml"
)
PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_ANKLE_COLLISION_XML = (
    PHONEBOT_FV2_TORQUE_XML_ROOT / "scene_joystick_rough_terrain_alter_v2_ankle_collision.xml"
)

CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_XML = CHOPSTICKBOT_XML_ROOT / "scene_joystick_flat_terrain.xml"
CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_XML = CHOPSTICKBOT_XML_ROOT / "scene_joystick_rough_terrain.xml"
CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_XML = (
    CHOPSTICKBOT_XML_ROOT / "scene_joystick_flat_terrain_alternative_imu.xml"
)
CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_XML = (
    CHOPSTICKBOT_XML_ROOT / "scene_joystick_rough_terrain_alternative_imu.xml"
)

CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_TORQUE_XML = (
    CHOPSTICKBOT_TORQUE_XML_ROOT / "scene_joystick_flat_terrain_alternative_imu.xml"
)
CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_TORQUE_XML = (
    CHOPSTICKBOT_TORQUE_XML_ROOT / "scene_joystick_rough_terrain_alternative_imu.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  """Map task name to a scene XML."""
  mapping = {
      "phonebot_flat_terrain": PHONEBOT_JOYSTICK_FLAT_TERRAIN_XML,
      "phonebot_rough_terrain": PHONEBOT_JOYSTICK_ROUGH_TERRAIN_XML,
      "phonebot_flat_terrain_alternative_imu": PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_XML,
      "phonebot_rough_terrain_alternative_imu": PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_XML,
      "phonebot_flat_terrain_alternative_imu_fv2": PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_XML,
      "phonebot_rough_terrain_alternative_imu_fv2": PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_XML,
      "phonebot_flat_terrain_alternative_imu_fv2_torque": PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_XML,
      "phonebot_rough_terrain_alternative_imu_fv2_torque": PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_XML,
      "phonebot_flat_terrain_alternative_imu_fv2_torque_ankle_collision": PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_ANKLE_COLLISION_XML,
      "phonebot_rough_terrain_alternative_imu_fv2_torque_ankle_collision": PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_ANKLE_COLLISION_XML,
      "chopstickbot_flat_terrain": CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_XML,
      "chopstickbot_rough_terrain": CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_XML,
      "chopstickbot_flat_terrain_alternative_imu": CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_XML,
      "chopstickbot_rough_terrain_alternative_imu": CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_XML,
      "chopstickbot_flat_terrain_alternative_imu_torque": CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_TORQUE_XML,
      "chopstickbot_rough_terrain_alternative_imu_torque": CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_TORQUE_XML,
  }
  if task_name not in mapping:
    raise KeyError(f"Unknown Codesign task '{task_name}'. Available: {list(mapping.keys())}")
  return mapping[task_name]


def phonebot_task_to_xml(task: str) -> epath.Path:
  if task == "flat_terrain":
    return PHONEBOT_JOYSTICK_FLAT_TERRAIN_XML
  if task == "rough_terrain":
    return PHONEBOT_JOYSTICK_ROUGH_TERRAIN_XML
  if task == "flat_terrain_alternative_imu":
    return PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_XML
  if task == "rough_terrain_alternative_imu":
    return PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_XML
  if task == "flat_terrain_alternative_imu_fv2":
    return PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_XML
  if task == "rough_terrain_alternative_imu_fv2":
    return PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_XML
  if task == "flat_terrain_alternative_imu_fv2_torque":
    return PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_XML
  if task == "rough_terrain_alternative_imu_fv2_torque":
    return PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_XML
  if task == "flat_terrain_alternative_imu_fv2_torque_ankle_collision":
    return PHONEBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_ANKLE_COLLISION_XML
  if task == "rough_terrain_alternative_imu_fv2_torque_ankle_collision":
    return PHONEBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_FV2_TORQUE_ANKLE_COLLISION_XML
  raise KeyError(f"Unknown phonebot task '{task}'.")


def chopstickbot_task_to_xml(task: str) -> epath.Path:
  if task == "flat_terrain":
    return CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_XML
  if task == "rough_terrain":
    return CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_XML
  if task == "flat_terrain_alternative_imu":
    return CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_XML
  if task == "rough_terrain_alternative_imu":
    return CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_XML
  if task == "flat_terrain_alternative_imu_torque":
    return CHOPSTICKBOT_JOYSTICK_FLAT_TERRAIN_ALTERNATIVE_IMU_TORQUE_XML
  if task == "rough_terrain_alternative_imu_torque":
    return CHOPSTICKBOT_JOYSTICK_ROUGH_TERRAIN_ALTERNATIVE_IMU_TORQUE_XML
  raise KeyError(f"Unknown chopstickbot task '{task}'.")


FEET_SITES = [
    "left_foot",
    "right_foot",
]

HAND_SITES = [
    "left_hand",
    "right_hand",
]

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "base_motor_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"