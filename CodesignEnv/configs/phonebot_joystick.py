"""Joystick training task for PhoneBot."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from . import hi_constants as consts
from ._joystick_common import BaseJoystick, default_config as _default_config


def default_config() -> config_dict.ConfigDict:
  cfg = _default_config()
  # Phonebot is taller; allow slightly higher commanded lateral speeds.
  cfg.lin_vel_y = [-0.8, 0.8]
  return cfg


class Joystick(BaseJoystick):
  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.phonebot_task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )


