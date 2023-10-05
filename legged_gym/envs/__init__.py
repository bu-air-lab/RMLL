from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .a1.rm.a1_rm_config import A1RMCfg, A1RMCfgPPO

import os

from legged_gym.utils.task_registry import task_registry


task_registry.register( "a1_rm", LeggedRobot, A1RMCfg(), A1RMCfgPPO() )
