"""
Vectorized RM, built for compatibility with Isaac Gym (maintain each envs RM state on GPU)
"""

import torch

class VecRewardMachine:

    def __init__(self, num_envs, device, bonus=1000):
        self.num_envs = num_envs
        self.rm_rews = torch.zeros(self.num_envs, device=device, dtype=torch.float)
        self.bonus=bonus


    def get_next_states(self, current_states, true_props, gait):

        next_states = torch.clone(current_states)

        if(gait == 'trot' or gait == 'pace' or gait == 'bound'):

            #Update from q0 -> q1 if true_props = 1
            q0_q1_indicies = (true_props == 1).nonzero()
            next_states[q0_q1_indicies] = 1

            #Update from q1 -> q0 if true_props = 2
            q1_q0_indicies = (true_props == 2).nonzero()
            next_states[q1_q0_indicies] = 0

        # elif(gait == 'biped_bound'):

        #     #Update from q0 -> q1 if true_props = 1
        #     q0_q1_indicies = (true_props == 1).nonzero()
        #     next_states[q0_q1_indicies] = 1

        #     #Update from q1 -> q2 if true_props = 2
        #     q1_q2_indicies = (true_props == 2).nonzero()
        #     next_states[q1_q2_indicies] = 2

        #     #Update from q2 -> q0 if true_props = 3
        #     q2_q3_indicies = (true_props == 3).nonzero()
        #     next_states[q2_q3_indicies] = 0

        #walk gaits are 4-state RMs
        elif(gait =='walk' or gait == 'three_one'):

            #Update from q0 -> q1 if true_props = 1
            q0_q1_indicies = (true_props == 1).nonzero()
            next_states[q0_q1_indicies] = 1

            #Update from q1 -> q2 if true_props = 2
            q1_q2_indicies = (true_props == 2).nonzero()
            next_states[q1_q2_indicies] = 2

            #Update from q2 -> q3 if true_props = 3
            q2_q3_indicies = (true_props == 3).nonzero()
            next_states[q2_q3_indicies] = 3

            #Update from q3 -> q0 if true_props = 4
            q3_q0_indicies = (true_props == 4).nonzero()
            next_states[q3_q0_indicies] = 0

        elif(gait == 'half_bound'):

            #Update from q0 -> q1 if true_props = 1
            q0_q1_indicies = (true_props == 1).nonzero()
            next_states[q0_q1_indicies] = 1

            #Update from q1 -> q2 if true_props = 2
            q1_q2_indicies = (true_props == 2).nonzero()
            next_states[q1_q2_indicies] = 2

            #Update from q2 -> q3 if true_props = 3
            q2_q3_indicies = (true_props == 3).nonzero()
            next_states[q2_q3_indicies] = 3

            #Update from q3 -> q0 if true_props = 4
            q3_q0_indicies = (true_props == 4).nonzero()
            next_states[q3_q0_indicies] = 0

            sink_state_indicies = (true_props == -1).nonzero()
            next_states[sink_state_indicies] = 4

        else:
            print("DEFINE REWARD IN vec_reward_machine.py")

        return next_states

    def get_reward(self, current_states, next_states, s_info, gait):

        #Populate rm_rews with all self-loop rewards
        self.rm_rews = s_info['computed_reward']

        #Find environments that had an RM transition, and replace rm_rews with bonus reward
        if(gait != 'half_bound'):
            bonus_envs = (current_states - next_states).nonzero()
            self.rm_rews[bonus_envs] *= self.bonus
        
        #For gaits which include sink state, give 0 reward when transitioning to or at sink state
        elif(gait == 'half_bound'):
            bonus_envs = (current_states - next_states).nonzero()
            self.rm_rews[bonus_envs] *= self.bonus

            sink_state_envs = (next_states == 4).nonzero()
            self.rm_rews[sink_state_envs] = 0
            #print(bonus_envs, sink_state_transition_envs)


    def step(self, current_states, true_props, s_info, gait):
        """
        Emulates a step on the reward machines from current_states when observing *true_props*.
        """

        # Computing the next RM state per env
        next_states = self.get_next_states(current_states, true_props, gait)

        #Update the reward
        self.get_reward(current_states, next_states, s_info, gait)

        return next_states, self.rm_rews

    #Bonus can be changed during training
    def set_bonus(self, bonus):
        self.bonus = bonus


