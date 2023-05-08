import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

def train(args):

	env, env_cfg = task_registry.make_env(name=args.task, args=args)
	ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
	ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
