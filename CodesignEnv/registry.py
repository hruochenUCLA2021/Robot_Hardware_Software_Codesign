"""
Registry for CodesignEnv environments.

Mirrors the structure of the reference HERMES HiEnv registry, but provides
tasks for PhoneBot and ChopstickBot joystick training.
"""

from .configs import chopstick_joystick
from .configs import chopstick_joystick_torque_awared
from .configs import phonebot_joystick
from .configs import phonebot_joystick_torque_awared
from .configs import phonebot_joystick_torque_awared_ankle_collision


ENVIRONMENTS = [
    "ChopstickbotJoystickFlatTerrain",
    "ChopstickbotJoystickRoughTerrain",
    "PhonebotJoystickFlatTerrain",
    "PhonebotJoystickRoughTerrain",
    "ChopstickbotJoystickFlatTerrainAlter",
    "ChopstickbotJoystickRoughTerrainAlter",
    "ChopstickbotJoystickFlatTerrainAlterTorqueAwared",
    "ChopstickbotJoystickRoughTerrainAlterTorqueAwared",
    "PhonebotJoystickFlatTerrainAlter",
    "PhonebotJoystickRoughTerrainAlter",
    "PhonebotJoystickFlatTerrainAlterFV2",
    "PhonebotJoystickRoughTerrainAlterFV2",
    "PhonebotJoystickFlatTerrainAlterFV2TorqueAwared",
    "PhonebotJoystickRoughTerrainAlterFV2TorqueAwared",
    "PhonebotJoystickFlatTerrainAlterFV2TorqueAwaredAnkleCollision",
    "PhonebotJoystickRoughTerrainAlterFV2TorqueAwaredAnkleCollision",
]


def get_environment(name: str):
    """Get (EnvClass, default_config_fn) by environment name."""
    if name == "ChopstickbotJoystickFlatTerrain":
        return chopstick_joystick.Joystick, chopstick_joystick.default_config
    if name == "ChopstickbotJoystickRoughTerrain":
        return chopstick_joystick.Joystick, chopstick_joystick.default_config
    if name == "PhonebotJoystickFlatTerrain":
        return phonebot_joystick.Joystick, phonebot_joystick.default_config
    if name == "PhonebotJoystickRoughTerrain":
        return phonebot_joystick.Joystick, phonebot_joystick.default_config
    if name == "ChopstickbotJoystickFlatTerrainAlter":
        return chopstick_joystick.Joystick, chopstick_joystick.default_config
    if name == "ChopstickbotJoystickRoughTerrainAlter":
        return chopstick_joystick.Joystick, chopstick_joystick.default_config
    if name == "ChopstickbotJoystickFlatTerrainAlterTorqueAwared":
        return chopstick_joystick_torque_awared.Joystick, chopstick_joystick_torque_awared.default_config
    if name == "ChopstickbotJoystickRoughTerrainAlterTorqueAwared":
        return chopstick_joystick_torque_awared.Joystick, chopstick_joystick_torque_awared.default_config
    if name == "PhonebotJoystickFlatTerrainAlter":
        return phonebot_joystick.Joystick, phonebot_joystick.default_config
    if name == "PhonebotJoystickRoughTerrainAlter":
        return phonebot_joystick.Joystick, phonebot_joystick.default_config
    if name == "PhonebotJoystickFlatTerrainAlterFV2":
        return phonebot_joystick.Joystick, phonebot_joystick.default_config
    if name == "PhonebotJoystickRoughTerrainAlterFV2":
        return phonebot_joystick.Joystick, phonebot_joystick.default_config
    if name == "PhonebotJoystickFlatTerrainAlterFV2TorqueAwared":
        return phonebot_joystick_torque_awared.Joystick, phonebot_joystick_torque_awared.default_config
    if name == "PhonebotJoystickRoughTerrainAlterFV2TorqueAwared":
        return phonebot_joystick_torque_awared.Joystick, phonebot_joystick_torque_awared.default_config
    if name == "PhonebotJoystickFlatTerrainAlterFV2TorqueAwaredAnkleCollision":
        return phonebot_joystick_torque_awared_ankle_collision.Joystick, phonebot_joystick_torque_awared_ankle_collision.default_config
    if name == "PhonebotJoystickRoughTerrainAlterFV2TorqueAwaredAnkleCollision":
        return phonebot_joystick_torque_awared_ankle_collision.Joystick, phonebot_joystick_torque_awared_ankle_collision.default_config

    raise ValueError(f"Unknown environment: {name}. Available: {ENVIRONMENTS}")


