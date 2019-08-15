# Functions to create training environment for each process

# Wrapper to modify rewards if needed (after each env.step(action))
# Football env has rewards -1 or 1
# Allowing only simple rewards makes the problem easier to solve, as Vlad Mnih states in this video:
# @1:31:50 https://youtu.be/-mhBD8Frkc4?t=5515
from gym import Wrapper
import numpy as np
import gfootball.env as football_env
class TSDiscountReward(Wrapper):
    """Time step discount reward:
        Adds negative reward after each step
    """
    def __init__(self, env=None, monitor=None):
        super(TSDiscountReward, self).__init__(env)
        #self.time_step_penalty = 1./500
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        #reward -= self.time_step_penalty
        #if reward > 0:
        #    reward = 1.
        #else:
        #    reward = 0.
        #:]("\naction:",action,"  reward:", reward)
        #print(info, "\n")
        return state, reward, done, info

    #def reset(self):
    #    #Do something at env.reset()

def create_train_env(layout, num_processes_to_render, index=None):
    if index is not None:
        # Called from asynchronous process with id==index
        print("Process {} - Create train env for {}".format(index, layout))
        if index < num_processes_to_render:
            # Only render first process
            render=True
        else:
            render=False
    else:
        # Called from global process, only to get data from env to create NN model
        print("Getting number of NN inputs/outputs for", layout)
        render=False
    #print(football_action_set.action_set_dict['default'])
    choose_random_env = False
    if choose_random_env:
        # Choose random env from all possibles variations
        chosen_env = np.random.choice(['academy_empty_goal_close',
                                   'academy_empty_goal',
                                   'academy_run_to_score',
                                   'academy_run_to_score_with_keeper',
                                   'academy_pass_and_shoot_with_keeper',
                                   'academy_run_pass_and_shoot_with_keeper',
                                   'academy_3_vs_1_with_keeper',
                                   'academy_corner',
                                   'academy_counterattack_easy',
                                   'academy_counterattack_hard',
                                   'academy_single_goal_versus_lazy'])
    else:
        chosen_env = 'academy_empty_goal_close'
    # Creating env for individual process
    env = football_env.create_environment(
                env_name=chosen_env, 
                stacked=False,                  # solo estado, no pixeles 
                representation='simple115',     # solo estado, no pixeles 
                rewards='scoring', # 1 point when score
                #rewards='scoring,checkpoints', # small scores <1 when closer to goal 
                render=render)                  # mostrar graficamente
    # Wrap around env:
    # Add penalty to each time step
    env = TSDiscountReward(env)
    num_inputs_to_nn = 115 # vector de 115 variables de estado para 'simple115'
    num_outputs_from_nn = env.action_space.n
    return env, num_inputs_to_nn, num_outputs_from_nn


