"""
Registry for CodesignEnv environments.

Mirrors the structure of the reference HERMES HiEnv registry, but provides
tasks for PhoneBot and ChopstickBot joystick training.
"""

from .configs import chopstick_joystick
from .configs import phonebot_joystick


ENVIRONMENTS = [
    "ChopstickbotJoystickFlatTerrain",
    "ChopstickbotJoystickRoughTerrain",
    "PhonebotJoystickFlatTerrain",
    "PhonebotJoystickRoughTerrain",
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

    raise ValueError(f"Unknown environment: {name}. Available: {ENVIRONMENTS}")


