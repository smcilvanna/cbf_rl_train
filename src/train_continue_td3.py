#!/usr/bin/env python3
import gymnasium as gym
import register_env  # need this so module registers the custom environment!
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
# from itertools import product
# import time
import numpy as np
import os
import rospy


class CustomCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, name_prefix='rl_model', verbose=1):
        super(CustomCheckpointCallback, self).__init__(save_freq, save_path, name_prefix, verbose)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save the model
            path = os.path.join(self.save_path, f'{self.name_prefix}_{self.n_calls}_steps')
            self.model.save(path)
            # Save the replay buffer
            replay_buffer_path = os.path.join(self.save_path, f'{self.name_prefix}_{self.n_calls}_replay_buffer')
            self.model.save_replay_buffer(replay_buffer_path)
            if self.verbose > 1:
                print(f"Saving model and replay buffer to {path}")
        return True

def train_td3(train_steps):    
    env = gym.make('cbf-train-gzros-td3')                                               # Create the environment
    model =         TD3.load("/home/user/husky/td3_models/final/td3_run5_model_23995", env=env)   # Load the model from the saved file
    model.load_replay_buffer("/home/user/husky/td3_models/final/td3_run5_replay_buffer_23995")    # Load the replay buffer
    
    checkpoint_callback = CustomCheckpointCallback(       # Custom callback to save the model and replay buffer during training in case of crash
        save_freq=100, 
        save_path='/home/user/husky/td3_models/',
        name_prefix=f"td3_run{run_id}_chkpt")

    model.learn(total_timesteps=train_steps,
                callback=checkpoint_callback,
                log_interval=1)                    
    
    fin_runs = model._n_updates                                                                                 # Number of training runs for file tag
    model.save(              f"/home/user/husky/td3_models/final/td3_run{run_id}_model_{fin_runs}")             # Save the model
    model.save_replay_buffer(f"/home/user/husky/td3_models/final/td3_run{run_id}_replay_buffer_{fin_runs}")     # Save the replay buffer

    return model

if __name__ == "__main__":
    run_id = 5                                      # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Run ID for file tag - CHANGE EVERY TRAINING SESSION !!!!!!!!!
    rospy.init_node('gymnode', anonymous=True)      # Initialize the node
    steps = 10000
    model = train_td3(train_steps=steps)
    print(">>>>>>>>>>>>>===============<<<<<<<<<<<<<<\n           TRAINING COMPLETE!             \n>>>>>>>>>>>>>===============<<<<<<<<<<<<<<")

