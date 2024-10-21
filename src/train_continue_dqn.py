#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import os
import rospy

##########################################################
from gymnasium.envs.registration import register
register(
    id='cbf-train-gzros-dqn',  # Unique identifier for the environment
    entry_point='envs.cbf_gzros_dqn:CustomEnv',  # Corrected path
    max_episode_steps=1,  # Maximum number of steps per episode
)
###########################################################

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

def train_dqn(train_steps):    
    env = gym.make('cbf-train-gzros-dqn')   # Create the environment
    maxActnp = np.load(loc_maxAct)          # Load max action array
    env.maxActPerObs = maxActnp             # Apply max action array to env
    
    model= DQN.load(loc_model, env=env)     # Load the model from the saved file
    model.load_replay_buffer(loc_buffer)    # Load the replay buffer
    
    checkpoint_callback = CustomCheckpointCallback(       # Custom callback to save the model and replay buffer during training in case of crash
        save_freq=100, 
        save_path='/home/user/husky/dqn_models/',
        name_prefix=f"dqn_run{run_id}_chkpt")

    model.learn(total_timesteps=train_steps,
                callback=checkpoint_callback,
                log_interval=1)                    
    
    fin_runs = model._n_updates                                                                                 # Number of training runs for file tag
    model.save(              f"/home/user/husky/dqn_models/final/dqn_run{run_id}_model_{fin_runs}")             # Save the model
    model.save_replay_buffer(f"/home/user/husky/dqn_models/final/dqn_run{run_id}_replay_buffer_{fin_runs}")     # Save the replay buffer

    return model

if __name__ == "__main__":
    run_id = 372                                      # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Run ID for file tag - CHANGE EVERY TRAINING SESSION !!!!!!!!!
    
    loc_model=  "/home/user/husky/dqn_models/final/dqn_run372_model"
    loc_buffer= "/home/user/husky/dqn_models/final/dqn_run372_replay_buffer"
    loc_maxAct= "/home/user/husky/dqn_models/final/dqn_run372_maxActs.npy"

    rospy.init_node('gymnode', anonymous=True)      # Initialize the node
    steps = 5000
    model = train_dqn(train_steps=steps)
    print(">>>>>>>>>>>>>===============<<<<<<<<<<<<<<\n           TRAINING COMPLETE!             \n>>>>>>>>>>>>>===============<<<<<<<<<<<<<<")

