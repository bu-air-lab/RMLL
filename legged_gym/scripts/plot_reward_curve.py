
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import matplotlib.pyplot as plt


args = get_args()
args.headless = True
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

# override some parameters for testing
env_cfg.env.num_envs = 10

env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.measure_heights = False
env_cfg.terrain.num_rows = 5
env_cfg.terrain.num_cols = 5
env_cfg.terrain.curriculum = False

env_cfg.noise.add_noise = False

env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False


#methods = ['rm', 'naive']
methods = ['rm']

rewards = []

num_saved_policies = 150
num_iters = int(num_saved_policies/10)
iter_amount = 100

for method in methods:

    train_cfg.runner.load_run = method
    args.experiment = method
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    method_rewards = []

    #Deploy every policy (saved every 5 iterations)
    for policy_iter in range(100, num_iters*iter_amount, iter_amount):

        # load policy
        train_cfg.runner.resume = True
        train_cfg.runner.checkpoint = policy_iter
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)

        #Deploy policy over 10 envs at once
        #Compute total reward accross all envs.
        reward = 0
        for i in range(int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            #reward += torch.sum(infos['non_RM_reward']).item()
            reward += torch.sum(rews).item()

        #Add avg reward a single policy achieved
        print(reward)
        method_rewards.append(reward/env_cfg.env.num_envs)

    rewards.append(method_rewards)

    del env


fig, ax = plt.subplots()
time = [i*iter_amount for i in range(num_iters)]

for i,method in enumerate(methods):
    ax.plot(time, rewards[i], label=method)

ax.legend()

plt.xlabel('Training Iterations')
plt.ylabel('Reward')
plt.savefig("plot.pdf", bbox_inches='tight')


"""for i,method in enumerate(methods):

    std_below = []
    std_above = []

    for t, reward in enumerate(rewards[i]):
        std_below.append(reward - reward_stds[i,t])
        std_above.append(reward + reward_stds[i,t])


    ax.plot(time, rewards[i], label=l)
    ax.fill_between(time, std_below, std_above, alpha=.1)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=13)

filename = version + '_reward_plot.pdf'

plt.xlabel('Training Iterations (in millions)')
plt.ylabel('Average Reward')

plt.yticks([x*100000 for x in [-1,0,1,2,3,4]], [-1, 0, 1, 2, 3, 4])
plt.xticks([x*10 for x in range(6)], [0, 1, 2, 3, 4, 5])

plt.savefig(filename, bbox_inches='tight')"""
