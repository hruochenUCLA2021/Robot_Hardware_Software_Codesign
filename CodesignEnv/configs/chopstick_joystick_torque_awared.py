"""Joystick training task for ChopstickBot."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from . import hi_constants as consts
from ._joystick_base_chopstickbot_torque_awared import BaseJoystick, default_config as _default_config


def default_config() -> config_dict.ConfigDict:
  cfg = _default_config()
  # ChopstickBot is simpler; keep smaller command ranges by default.
  cfg.lin_vel_x = [-0.6, 0.6]
  cfg.lin_vel_y = [-0.4, 0.4]
  cfg.ang_vel_yaw = [-0.8, 0.8]
  return cfg


class Joystick(BaseJoystick):
  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # Allow training/rollout scripts to override the exact scene XML path.
    xml_override = ""
    try:
      xml_override = str(getattr(config, "xml_path_override", "") or "").strip()
    except Exception:
      xml_override = ""

    if xml_override:
      # Resolve relative paths against Robot_Hardware_Software_Codesign/ root.
      try:
        from etils import epath

        p = epath.Path(xml_override).expanduser()
        if not p.is_absolute():
          p = (consts.PROJECT_ROOT / p).resolve()
        xml_path = p.as_posix()
      except Exception:
        xml_path = xml_override
    else:
      xml_path = consts.chopstickbot_task_to_xml(task).as_posix()
    super().__init__(
        xml_path=xml_path,
        config=config,
        config_overrides=config_overrides,
    )


