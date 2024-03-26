from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import random

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
#from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.base_RM_task import BaseRMTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseRMTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, gait, seed):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.gait = gait
        self.seed = seed
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.gait)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()

        #Initialize masks needed for evaluation of propositional symbols truth values
        #TODO: There's probably a cleaner way of doing this
        #Foot order: FL, FR, RL, RR
        self.FL_RR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FL_RR_mask[:,0] = 1
        self.FL_RR_mask[:,3] = 1

        self.FR_RL_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FR_RL_mask[:,1] = 1
        self.FR_RL_mask[:,2] = 1

        self.FL_RL_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FL_RL_mask[:,0] = 1
        self.FL_RL_mask[:,2] = 1

        self.FR_RR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FR_RR_mask[:,1] = 1
        self.FR_RR_mask[:,3] = 1

        self.FL_FR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FL_FR_mask[:,0] = 1
        self.FL_FR_mask[:,1] = 1

        self.RL_RR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.RL_RR_mask[:,2] = 1
        self.RL_RR_mask[:,3] = 1

        self.FL_FR_RL_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FL_FR_RL_mask[:,0] = 1
        self.FL_FR_RL_mask[:,1] = 1
        self.FL_FR_RL_mask[:,2] = 1

        self.FL_FR_RR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FL_FR_RR_mask[:,0] = 1
        self.FL_FR_RR_mask[:,1] = 1
        self.FL_FR_RR_mask[:,3] = 1

        self.FL_RL_RR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FL_RL_RR_mask[:,0] = 1
        self.FL_RL_RR_mask[:,2] = 1
        self.FL_RL_RR_mask[:,3] = 1

        self.FR_RL_RR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FR_RL_RR_mask[:,1] = 1
        self.FR_RL_RR_mask[:,2] = 1
        self.FR_RL_RR_mask[:,3] = 1

        self.FL_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FL_mask[:,0] = 1

        self.FR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.FR_mask[:,1] = 1

        self.RL_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.RL_mask[:,2] = 1

        self.RR_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.RR_mask[:,3] = 1

        self.none_mask = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.all_mask = torch.ones(self.num_envs, 4, device=self.device, dtype=torch.long)

        self.init_done = True

    #Get intersection between 2 tensors
    #See https://discuss.pytorch.org/t/intersection-between-to-vectors-tensors/50364/9
    def intersection(self, first, second):
        return first[(first.view(1, -1) == second.view(-1, 1)).any(dim=0)]

    #Returns environments that contain specified foot contacts
    #Input a mask for contacts we want, and another mask for contacts we dont
    def contact_envs(self, foot_contacts, wanted_contacts_mask, unwanted_contacts_mask):

        #Number of feet we want to make contact
        num_wanted_contacts = torch.sum(wanted_contacts_mask, dim=1)[0].item()

        #Filter out environments based on which have exclusively wanted feet contacts or exclusively unwanted contacts
        wanted_contacts_masked = foot_contacts*wanted_contacts_mask
        unwanted_contacts_masked = foot_contacts*unwanted_contacts_mask

        #Find environments which have wanted contacts
        wanted_contacts = (torch.sum(wanted_contacts_masked, dim=1) == num_wanted_contacts).nonzero()
        unwanted_contacts = (torch.sum(unwanted_contacts_masked, dim=1) > 0).nonzero()

        #Remove environments with unwanted contacts from environments containing all wanted contacts
        matching_contacts = self.intersection(wanted_contacts, unwanted_contacts)

        for item in matching_contacts:
            wanted_contacts = wanted_contacts[wanted_contacts != item[0]]

        return wanted_contacts


    #Needed for RM implementation
    #Returns the truth value of all propositional symbols, for each environment
    def get_events(self):

        prop_symbols = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        #Foot order: FL, FR, RL, RR
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_contacts = torch.logical_or(contact, self.last_contacts)

        #Environment ids of each state
        q0_envs = (self.current_rm_states_buf == 0).nonzero()
        q1_envs = (self.current_rm_states_buf == 1).nonzero()        
        q2_envs = (self.current_rm_states_buf == 2).nonzero()
        q3_envs = (self.current_rm_states_buf == 3).nonzero()
        q4_envs = (self.current_rm_states_buf == 4).nonzero()
        q5_envs = (self.current_rm_states_buf == 5).nonzero()

        if(self.gait == 'trot'):

            FL_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_RR_mask,
                                                unwanted_contacts_mask = self.FR_RL_mask)
            FL_RR_contacts_q0 = self.intersection(FL_RR_contacts, q0_envs)
            q0_q1_envs = self.intersection(FL_RR_contacts_q0, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q0_q1_envs] = 1


            FR_RL_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FR_RL_mask,
                                                unwanted_contacts_mask = self.FL_RR_mask)
            FR_RL_contacts_q1 = self.intersection(FR_RL_contacts, q1_envs)
            q1_q0_envs = self.intersection(FR_RL_contacts_q1, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q1_q0_envs = self.intersection(q1_q0_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            q1_q0_envs = self.intersection(q1_q0_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())

            prop_symbols[q1_q0_envs] = 2


        elif(self.gait == 'pace'):

            FL_RL_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_RL_mask,
                                                unwanted_contacts_mask = self.FR_RR_mask)
            FL_RL_contacts_q0 = self.intersection(FL_RL_contacts, q0_envs)
            q0_q1_envs = self.intersection(FL_RL_contacts_q0, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q0_q1_envs] = 1


            FR_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FR_RR_mask,
                                                unwanted_contacts_mask = self.FL_RL_mask)
            FR_RR_contacts_q1 = self.intersection(FR_RR_contacts, q1_envs)
            q1_q0_envs = self.intersection(FR_RR_contacts_q1, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q1_q0_envs = self.intersection(q1_q0_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            q1_q0_envs = self.intersection(q1_q0_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q1_q0_envs] = 2


        elif(self.gait == 'bound'):

            FL_FR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_FR_mask,
                                                unwanted_contacts_mask = self.RL_RR_mask)
            FL_FR_contacts_q0 = self.intersection(FL_FR_contacts, q0_envs)
            q0_q1_envs = self.intersection(FL_FR_contacts_q0, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q0_q1_envs] = 1


            RL_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.RL_RR_mask,
                                                unwanted_contacts_mask = self.FL_FR_mask)
            RL_RR_contacts_q1 = self.intersection(RL_RR_contacts, q1_envs)
            q1_q0_envs = self.intersection(RL_RR_contacts_q1, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q1_q0_envs = self.intersection(q1_q0_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            q1_q0_envs = self.intersection(q1_q0_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q1_q0_envs] = 2


        elif(self.gait == 'walk'):

            FR_RL_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FR_RL_RR_mask,
                                                unwanted_contacts_mask = self.FL_mask)
            FR_RL_RR_contacts_q0 = self.intersection(FR_RL_RR_contacts, q0_envs)
            q0_q1_envs = self.intersection(FR_RL_RR_contacts_q0, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)[:]).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q0_q1_envs] = 1


            FL_FR_RL_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_FR_RL_mask,
                                                unwanted_contacts_mask = self.RR_mask)
            FL_FR_RL_contacts_q1 = self.intersection(FL_FR_RL_contacts, q1_envs)
            q1_q2_envs = self.intersection(FL_FR_RL_contacts_q1, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)[:]).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q1_q2_envs] = 2


            FL_RL_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_RL_RR_mask,
                                                unwanted_contacts_mask = self.FR_mask)
            FL_RL_RR_contacts_q2 = self.intersection(FL_RL_RR_contacts, q2_envs)
            q2_q3_envs = self.intersection(FL_RL_RR_contacts_q2, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)[:]).nonzero())
            q2_q3_envs = self.intersection(q2_q3_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q2_q3_envs] = 3


            FL_FR_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_FR_RR_mask,
                                                unwanted_contacts_mask = self.RL_mask)
            FL_FR_RR_contacts_q3 = self.intersection(FL_FR_RR_contacts, q3_envs)
            q3_q0_envs = self.intersection(FL_FR_RR_contacts_q3, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)[:]).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q3_q0_envs] = 4


        #FL/RR/RL -> FL -> FR/RL/RR -> FR -> ...
        elif(self.gait == 'three_one'):

            FL_RL_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_RL_RR_mask,
                                                unwanted_contacts_mask = self.FR_mask)
            FL_RL_RR_contacts_q0 = self.intersection(FL_RL_RR_contacts, q0_envs)
            q0_q1_envs = self.intersection(FL_RL_RR_contacts_q0, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q0_q1_envs] = 1

            FL_contacts = self.contact_envs(foot_contacts, 
                        wanted_contacts_mask=self.FL_mask,
                        unwanted_contacts_mask = self.FR_RL_RR_mask)
            FL_contacts_q1 = self.intersection(FL_contacts, q1_envs)
            q1_q2_envs = self.intersection(FL_contacts_q1, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)/2).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())

            prop_symbols[q1_q2_envs] = 2

            FR_RL_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FR_RL_RR_mask,
                                                unwanted_contacts_mask = self.FL_mask)
            FR_RL_RR_contacts_q2 = self.intersection(FR_RL_RR_contacts, q2_envs)
            q2_q3_envs = self.intersection(FR_RL_RR_contacts_q2, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q2_q3_envs = self.intersection(q2_q3_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q2_q3_envs] = 3

            FR_contacts = self.contact_envs(foot_contacts, 
                                    wanted_contacts_mask=self.FR_mask,
                                    unwanted_contacts_mask = self.FL_RL_RR_mask)
            FR_contacts_q3 = self.intersection(FR_contacts, q3_envs)
            q3_q0_envs = self.intersection(FR_contacts_q3, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)/2).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q3_q0_envs] = 4


        elif(self.gait == 'half_bound'):


            FL_FR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_FR_mask,
                                                unwanted_contacts_mask = self.RL_RR_mask)
            FL_FR_contacts_q0 = self.intersection(FL_FR_contacts, q0_envs)
            q0_q1_envs = self.intersection(FL_FR_contacts_q0, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q0_q1_envs] = 1


            RR_contacts = self.contact_envs(foot_contacts, 
                                    wanted_contacts_mask=self.RR_mask,
                                    unwanted_contacts_mask = self.FL_FR_RL_mask)
            RR_contacts_q1 = self.intersection(RR_contacts, q1_envs)
            q1_q2_envs = self.intersection(RR_contacts_q1, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q1_q2_envs] = 2


            FL_FR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_FR_mask,
                                                unwanted_contacts_mask = self.RL_RR_mask)
            FL_FR_contacts_q2 = self.intersection(FL_FR_contacts, q2_envs)
            q2_q3_envs = self.intersection(FL_FR_contacts_q2, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q2_q3_envs = self.intersection(q2_q3_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            q2_q3_envs = self.intersection(q2_q3_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q2_q3_envs] = 3


            RL_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.RL_mask,
                                                unwanted_contacts_mask = self.FL_FR_RR_mask)
            RL_contacts_q3 = self.intersection(RL_contacts, q3_envs)
            q3_q0_envs = self.intersection(RL_contacts_q3, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q3_q0_envs] = 4

            #Go to sink state if RL makes contact while at state q1 or q2
            #Go to sink state if RR makes contact while at state q3 or q0
            RL_contact_envs = foot_contacts[:,2].nonzero()
            RR_contact_envs = foot_contacts[:,3].nonzero()

            RL_contact_envs_q1 = self.intersection(RL_contact_envs, q1_envs)
            RL_contact_envs_q2 = self.intersection(RL_contact_envs, q2_envs)

            RR_contact_envs_q3 = self.intersection(RR_contact_envs, q3_envs)
            RR_contact_envs_q0 = self.intersection(RR_contact_envs, q0_envs)

            #Give 20 step grace period in begginning of episode where RR foot can make contact
            RR_contact_envs_q0 = self.intersection(RR_contact_envs_q0, (self.episode_length_buf[:] >= 20).nonzero())

            prop_symbols[RL_contact_envs_q1] = -1
            prop_symbols[RL_contact_envs_q2] = -1
            prop_symbols[RR_contact_envs_q3] = -1
            prop_symbols[RR_contact_envs_q0] = -1

        #RL -> FL/RR -> FL/FR/RR -> FR -> None -> RL -> ....
        elif(self.gait == 'canter'):


            RL_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.RL_mask,
                                                unwanted_contacts_mask = self.FL_FR_RR_mask)
            RL_contacts_q0 = self.intersection(RL_contacts, q0_envs)
            q0_q1_envs = self.intersection(RL_contacts_q0, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            q0_q1_envs = self.intersection(q0_q1_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q0_q1_envs] = 1


            FL_RR_contacts = self.contact_envs(foot_contacts, 
                                    wanted_contacts_mask=self.FL_RR_mask,
                                    unwanted_contacts_mask = self.FR_RL_mask)
            FL_RR_contacts_q1 = self.intersection(FL_RR_contacts, q1_envs)
            q1_q2_envs = self.intersection(FL_RR_contacts_q1, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            q1_q2_envs = self.intersection(q1_q2_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q1_q2_envs] = 2

            #Add sink state to avoid RL contact here?
            FL_FR_RR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FL_FR_RR_mask,
                                                unwanted_contacts_mask = self.RL_mask)
            FL_FR_RR_contacts_q2 = self.intersection(FL_FR_RR_contacts, q2_envs)
            q2_q3_envs = self.intersection(FL_FR_RR_contacts_q2, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q2_q3_envs = self.intersection(q2_q3_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q2_q3_envs] = 3


            FR_contacts = self.contact_envs(foot_contacts, 
                                                wanted_contacts_mask=self.FR_mask,
                                                unwanted_contacts_mask = self.FL_RL_RR_mask)
            FR_contacts_q3 = self.intersection(FR_contacts, q3_envs)
            # q3_q4_envs = self.intersection(FR_contacts_q3, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            # q3_q4_envs = self.intersection(q3_q4_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            # q3_q4_envs = self.intersection(q3_q4_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            # q3_q4_envs = self.intersection(q3_q4_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            # prop_symbols[q3_q4_envs] = 4
            q3_q0_envs = self.intersection(FR_contacts_q3, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            q3_q0_envs = self.intersection(q3_q0_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            prop_symbols[q3_q0_envs] = 4

            # none_contacts = self.contact_envs(foot_contacts, 
            #                                     wanted_contacts_mask=self.none_mask,
            #                                     unwanted_contacts_mask = self.all_mask)
            # none_contacts_q4 = self.intersection(none_contacts, q4_envs)
            # q4_q0_envs = self.intersection(none_contacts_q4, (self.rm_iters[:] >= self.commanded_gait_freq.squeeze(1)).nonzero())
            # q4_q0_envs = self.intersection(q4_q0_envs, (self.foot_heights[:,0] >= self.cfg.env.min_foot_height).nonzero())
            # q4_q0_envs = self.intersection(q4_q0_envs, (self.foot_heights[:,1] >= self.cfg.env.min_foot_height).nonzero())
            # q4_q0_envs = self.intersection(q4_q0_envs, (self.foot_heights[:,2] >= self.cfg.env.min_foot_height).nonzero())
            # q4_q0_envs = self.intersection(q4_q0_envs, (self.foot_heights[:,3] >= self.cfg.env.min_foot_height).nonzero())
            # prop_symbols[q4_q0_envs] = 5



            #Go to sink state if FL makes contact while at state q0 or q4
            #Go to sink state if FR makes contact while at state q0 or q1
            #Go to sink state if RL makes contact while at state q1 or q2 or q3 or q4
            #Go to sink state if RR makes contact while at state q0 or q4
            FL_contact_envs = foot_contacts[:,0].nonzero()
            FR_contact_envs = foot_contacts[:,1].nonzero()
            RL_contact_envs = foot_contacts[:,2].nonzero()
            RR_contact_envs = foot_contacts[:,3].nonzero()

            FL_contact_envs_q0 = self.intersection(FL_contact_envs, q0_envs)
            FL_contact_envs_q4 = self.intersection(FL_contact_envs, q4_envs)

            FR_contact_envs_q0 = self.intersection(FR_contact_envs, q0_envs)
            FR_contact_envs_q1 = self.intersection(FR_contact_envs, q1_envs)

            RL_contact_envs_q1 = self.intersection(RL_contact_envs, q1_envs)
            RL_contact_envs_q2 = self.intersection(RL_contact_envs, q2_envs)
            RL_contact_envs_q3 = self.intersection(RL_contact_envs, q3_envs)
            RL_contact_envs_q4 = self.intersection(RL_contact_envs, q4_envs)

            RR_contact_envs_q0 = self.intersection(RR_contact_envs, q0_envs)
            RR_contact_envs_q4 = self.intersection(RR_contact_envs, q4_envs)

            #Give 20 step grace period in beginning of episode where FL/FR/RR feet can make contact
            FL_contact_envs_q0 = self.intersection(FL_contact_envs_q0, (self.episode_length_buf[:] >= 20).nonzero())
            FR_contact_envs_q0 = self.intersection(FR_contact_envs_q0, (self.episode_length_buf[:] >= 20).nonzero())
            RR_contact_envs_q0 = self.intersection(RR_contact_envs_q0, (self.episode_length_buf[:] >= 20).nonzero())

            prop_symbols[FL_contact_envs_q0] = -1
            #prop_symbols[FL_contact_envs_q4] = -1

            prop_symbols[FR_contact_envs_q0] = -1
            prop_symbols[FR_contact_envs_q1] = -1

            prop_symbols[RL_contact_envs_q1] = -1
            prop_symbols[RL_contact_envs_q2] = -1
            prop_symbols[RL_contact_envs_q3] = -1
            #prop_symbols[RL_contact_envs_q4] = -1

            prop_symbols[RR_contact_envs_q0] = -1
            #prop_symbols[RR_contact_envs_q4] = -1


        return prop_symbols


    #Returns the approximate height of ground under each foot
    #Consider the average heights of four corners of measured_heights
    def ground_under_foot_heights(self):

        #Indicies in self.measured_heights which correspond to ground under each foot
        RR_indicies = [0, 1, 2, 11, 12, 13, 22, 23, 24]
        RL_indicies = [8, 9, 10, 19, 20, 21, 30, 31, 32]
        FR_indicies = [-11, -10, -9, -22, -21, -20, -33, -32, -31]
        FL_indicies = [-3, -2, -1, -14, -13, -12, -25, -24, -23]

        RR_ground_height = torch.mean(self.measured_heights[:,RR_indicies]).item()
        RL_ground_height = torch.mean(self.measured_heights[:,RL_indicies]).item()
        FR_ground_height = torch.mean(self.measured_heights[:,FR_indicies]).item()
        FL_ground_height = torch.mean(self.measured_heights[:,FL_indicies]).item()

        return RR_ground_height, RL_ground_height, FR_ground_height, FL_ground_height

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):

            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        #Compute rewards, update observation buffer (including RM state), etc...
        self.post_physics_step()


        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        self.extras['reward_components'] = self.reward_components

        #Needed for energy consumption calculation in scripts/measure_energy.py
        self.extras['torques'] = self.torques
        self.extras['dof_vels'] = self.dof_vel
        self.extras['lin_vel'] = self.base_lin_vel[:, :2]

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        #Needed to get foot coordinates
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        #Order: FL,FR,RL,RR
        self.foot_heights = self.link_positions[:, self.feet_indices, 2] #Global foot height

        #Subtract ground height below foot from foot height.
        #This is approximately how high the foot is in the air
        if(type(self.measured_heights) != int):

            RR_ground_height, RL_ground_height, FR_ground_height, FL_ground_height = self.ground_under_foot_heights()

            self.foot_heights[:,0] -= FL_ground_height
            self.foot_heights[:,1] -= FR_ground_height
            self.foot_heights[:,2] -= RL_ground_height
            self.foot_heights[:,3] -= RR_ground_height

        #Clip footheight minimum to 0 (foot is never below ground)
        self.foot_heights = torch.clamp(self.foot_heights, min=0)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        #Update RM states, return reward from RM
        true_props = self.get_events()
        info = {'computed_reward': self.rew_buf}
        new_rm_states, rm_rew = self.reward_machine.step(self.current_rm_states_buf, 
                                                        true_props, 
                                                        info, 
                                                        self.gait)

        #Add 1 to rm_iters per each env. 
        #Reset rm_iters for envs when RM state changes.
        changed_envs = (self.current_rm_states_buf - new_rm_states).nonzero()
        self.rm_iters[:] += 1
        self.rm_iters[changed_envs] = 0

        self.current_rm_states_buf = new_rm_states
        self.rew_buf = rm_rew

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """

        #Terminate on non-foot contact or timeout
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        #Terminate if joint position leaves bounds defined in URDF
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.) # upper limit
        out_of_limits = torch.sum(out_of_limits, dim=1)
        out_of_limits_envs = out_of_limits.nonzero()
        self.reset_buf[out_of_limits_envs] = True

        #Terminate if base height goes below some threshold
        # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # low_base_height_envs = (base_height < self.cfg.env.min_base_height).nonzero()
        # self.reset_buf[low_base_height_envs] = True

        #Terminate on excessive action differences. For some reason, action rate continuously increases otherwise.
        action_rates = self._reward_action_rate()
        excessive_action_rate_envs = (action_rates > self.cfg.env.max_action_rate).nonzero()
        self.reset_buf[excessive_action_rate_envs] = True

        #Terminate if foot swings above some threshold
        high_foot_height_envs = (self.foot_heights > self.cfg.env.max_foot_height).nonzero()[:,0]
        self.reset_buf[high_foot_height_envs] = True


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        #self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        #reset RMs indexed by env_ids
        self.current_rm_states_buf[env_ids] = 0
        self.rm_iters[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.reward_components[name] = rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """

        rm_state_encoding = F.one_hot(self.current_rm_states_buf, num_classes=self.num_rm_states)
        #print("RM state:", rm_state_encoding)
        self.obs_buf = torch.cat((
                            self.vel_commands[:, :2] * self.vel_commands_scale,
                            self.commanded_gait_freq * self.obs_scales.rm_iters_scale,
                            #self.commanded_base_height,
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            self.dof_vel * self.obs_scales.dof_vel,
                            self.actions,
                            #self.projected_gravity,
                            rm_state_encoding,
                            self.rm_iters.unsqueeze(1) * self.obs_scales.rm_iters_scale,
                            torch.zeros(self.num_envs, self.cfg.env.estimated_state_size, device=self.device, dtype=torch.float) #placeholder for estimated state
                            ),dim=-1)

        #base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1).unsqueeze(1)
        
        self.privileged_obs_buf = torch.cat((
                            self.vel_commands[:, :2] * self.vel_commands_scale,
                            self.commanded_gait_freq * self.obs_scales.rm_iters_scale,
                            #self.commanded_base_height,
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            self.dof_vel * self.obs_scales.dof_vel,
                            self.actions,
                            #self.projected_gravity,
                            rm_state_encoding,
                            self.rm_iters.unsqueeze(1) * self.obs_scales.rm_iters_scale,
                            #Only select x lin vel and ang vel
                            torch.index_select(self.base_lin_vel, 1, torch.cuda.LongTensor([0,2], device=self.device)) * self.obs_scales.lin_vel,
                            self.foot_heights#,
                            #base_height
                            ),dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    #After episode terminates, resample all commands
    #Also resample commands after "resampling_time" is reached
    def _resample_commands(self, env_ids):

        #Resample gait frequency commands
        self.commanded_gait_freq[env_ids] = torch.randint(self.cfg.commands.ranges.gait_freq_range[0], self.cfg.commands.ranges.gait_freq_range[1], (len(env_ids), 1), device=self.device)

        #self.commanded_base_height[env_ids, 0] = torch_rand_float(self.cfg.commands.ranges.base_height_range[0], self.cfg.commands.ranges.base_height_range[1], (len(env_ids), 1), device=self.device).squeeze(1)

        #Resample linear and angular velocity commands
        self.vel_commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.vel_commands[env_ids, 1] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.vel_commands[env_ids, :1] *= (torch.norm(self.vel_commands[env_ids, :1], dim=1) > 0.2).unsqueeze(1) # set small commands to zero

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        #actions_scaled = actions * self.action_scale

        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        #move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        move_down = (distance < torch.norm(self.vel_commands[env_ids, :1], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)

    def _update_action_scale_curriculum(self, env_ids):
        """
            Implements a curriculum of increasing range of motion for hip joints (indexed: 0,3,6,9)
        """
        # If the tracking reward is above 80% of the maximum, increase the range of motion
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:

            #Max action scale is 0.25
            if(self.action_scale[0] < 0.25):
                self.action_scale[[0,3,6,9]] += 0.01

    def _update_dof_acc_curriculum(self, env_ids):
        """
            Implements a curriculum of increasing joint acceleration penalty
        """
        # If the tracking reward is above 80% of the maximum, increase the penalty for joint acceleration. Starts at -2.5e-7
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.reward_scales['dof_acc'] -= (1e-7 * self.dt)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        #With base height tracking
        # noise_vec[:4] = 0. #Commands
        # noise_vec[4:16] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[16:28] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[28:] = 0. # previous actions + RM state + state estimation

        #Without base height tracking
        noise_vec[:3] = 0. #Commands
        noise_vec[3:15] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[15:27] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[27:] = 0. # previous actions + RM state + state estimation

        return noise_vec




    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """


        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        #Added to measure foot heights
        link_positions = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.link_state = gymtorch.wrap_tensor(link_positions)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.link_positions = self.link_state.view(self.num_envs, -1, 13)


        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        #Commanded linear and angular vel
        self.vel_commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.vel_commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)

        #Commanded gait frequency (number of timesteps between RM state changes)
        self.commanded_gait_freq = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False)

        #Commanded base height
        #self.commanded_base_height = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        #init all gait freq commands randomly within range
        self.commanded_gait_freq[:] = torch.randint(self.cfg.commands.ranges.gait_freq_range[0], self.cfg.commands.ranges.gait_freq_range[1], (self.num_envs, 1), device=self.device)
        #self.commanded_base_height[:] = torch_rand_float(self.cfg.commands.ranges.base_height_range[0], self.cfg.commands.ranges.base_height_range[1], (self.num_envs, 1), device=self.device)

        #self.vel_commands[env_ids, 1] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

        self.reward_components = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     #print("Base Height:", base_height)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.vel_commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_tracking_base_height(self):
    #     # Tracking of base height
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     base_height_error = torch.square(self.commanded_base_height[:,0] - base_height)
    #     return base_height_error


    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     #print("Base Height:", base_height)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.vel_commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.vel_commands[:, :1], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

