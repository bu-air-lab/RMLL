import math
import torch
import numpy as np

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c

#Bonus reward for taking an RM transition
#Scaled by velocity tracking reward, to discourage higher gait frequency
class RMTransitionReward(RewardFunction):

    def __init__(self, bonus):
        super().__init__()
        self.bonus = bonus

    def get_type(self):
        return "transition"

    #Only get a large bonus when we move far forward in x-direction
    def get_reward(self, s_info):

        #print("Bonus reward:", RMSelfLoopReward().get_reward(s_info)*self.bonus)
        #return RMSelfLoopReward().get_reward(s_info)*self.bonus

        return s_info['velocity_tracking_reward']*self.bonus


#Use base environment reward
#Encourages robot to follow velocity command, penalize energy, etc...
class RMSelfLoopReward(RewardFunction):

    def __init__(self):
        super().__init__()

    def get_type(self):
        return "self_loop"

    #Balance reward for moving in x-direction while minimizing movement in y-direction and energy consumption
    def get_reward(self, s_info):

        #print("Reward:", s_info['computed_reward'])
        return s_info['computed_reward']
