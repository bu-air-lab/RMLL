We modified this original repo: https://github.com/leggedrobotics/legged_gym, to add training via RM

Our RM implementation inspired by original RM code from https://github.com/RodrigoToroIcarte/reward_machines
Our implementation is compatible with vectorized environments on GPU

# Installation

1. Install pytorch with cuda and Isaac Gym (see https://github.com/leggedrobotics/legged_gym for details)
2. Install rm_ppo: ``` cd rm_ppo && pip3 install -e .```
3. Install legged_gym: ``` cd RMLL && pip3 install -e .```

# Train

Gait types: trot, bound, pace, walk, three_one, half_bound\
Experiment types: rm, noRM_history, noRM_foot_contacts, noRM


Example command to train pace gait, with rm state included in state space, on random seed 18:

```
python3 legged_gym/scripts/train.py --task=a1_rm --gait=trot --seed=18 --headless
```

-- headless means training will not be visualized.


# Validate:

Sim-to-sim transfer to PyBullet:

```
cd pybullet_val
python3 -m scripts.play_bullet.py
```

Be sure to update path to model in pybullet_val/scripts/play_bullet.py

# Test:

First update load_run and checkpoint in config file. Then, run:

```
python3 legged_gym/scripts/play.py --task=a1_rm --gait=trot --seed=0
```
