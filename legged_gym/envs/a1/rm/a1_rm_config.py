from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1RMCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 25 #1500 #4096
        num_observations = 47
        num_privileged_obs = 47

        min_foot_height = 0.03
        max_foot_height = 0.25

        max_action_rate = 40

    class commands (LeggedRobotCfg.commands):

        heading_command = False #Directly sample angular velocity command

        class ranges (LeggedRobotCfg.commands.ranges):

            lin_vel_x = [-1, 1] # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]

            gait_freq_range = [6, 12]
            base_height_range = [0.1, 0.3]

            #Only for walk gait
            # lin_vel_x = [-0.5, 0.5] # min max [m/s]
            # ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]

    class terrain( LeggedRobotCfg.terrain ):
        #mesh_type = 'plane'
        measure_heights = True

        #measure_heights = True
        #measured_points_x = [0.]
        #measured_points_y = [0.]

        curriculum = True
        terrain_proportions = [0, 1.0, 0, 0, 0]
        selected = False

        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)

    class sim( LeggedRobotCfg.sim ):
        dt = 0.005

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        #penalize_contacts_on = ["thigh", "calf"]
        #terminate_after_contacts_on = ["base"]
        terminate_after_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        dof_acc_curriculum = False
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_base_height = 0.5

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class noise ( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales ( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.25

class A1RMCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):


        #policy_class_name = 'ActorCriticRecurrent'
        policy_class_name = 'ActorCritic'

        run_name = ''
        experiment_name = 'rm_a1'
        max_iterations = 1000 # number of policy updates
        load_run = 'rm_pace18' # folder directly containing model files
        checkpoint = 200 # saved model iter
