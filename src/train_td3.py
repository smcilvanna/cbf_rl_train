#!/usr/bin/env python3
import gymnasium as gym
import register_env  # need this so module registers the custom environment!
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
# from itertools import product
# import time
# import numpy as np
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

def train_td3(learning_rate, gamma, batch_size, train_steps=100):    
    env = gym.make('cbf-train-gzros-td3')          # Create the environment
    model = TD3(                                   # Define the TD3 model
        "MlpPolicy",                                # Policy type (multi-layer perceptron)
        env,                                        # the custom environment
        verbose=                1,                  # Verbosity mode
        learning_rate=          learning_rate,      # Learning rate passed as an argument
        gamma=                  gamma,              # Discount factor passed as an argument
        batch_size=             batch_size,         # Batch size for training
        buffer_size=            1000,               # Replay buffer size
        learning_starts=        1,                  # Number of steps before training starts
        policy_delay=           1,                  # Policy update delay 1=update immediately (no delay in reward in this scenario)
        action_noise=           None,               # Action noise (minimal noise)
        target_policy_noise=    0.0,
        policy_kwargs=          dict(net_arch=[64, 64]),    # Policy network architecture
        tensorboard_log="/home/user/husky/tensorboard/td3", # Path to the directory where TensorBoard logs will be saved, uncomment for logging
        tau=                    0.05                        # Target network update coefficient, slightly larger for deterministic environment
    )

    checkpoint_callback = CustomCheckpointCallback(       # Callback to save the model and replay buffer every 10 steps/episodes
        save_freq=100, 
        save_path='/home/user/husky/td3_models/',
        name_prefix=f"td3_run{run_id}_chkpt"
    )

    # Train the model
    model.learn(total_timesteps=train_steps,
                callback=checkpoint_callback,
                log_interval=1)                    # train the model
    
    model.save(f"/home/user/husky/td3_models/final/td3_run{run_id}_model_{fin_runs}")                           # Save the model
    model.save_replay_buffer(f"/home/user/husky/td3_models/final/td3_run{run_id}_replay_buffer_{fin_runs}")     # Save the replay buffer

    return model

if __name__ == "__main__":
    
    run_id = 4
    fin_runs = 2000
    rospy.init_node('gymnode', anonymous=True)      # Initialize the node

    learnrate = 1e-3
    gamma     = 0.00
    batch     = 1
    steps = 2000
    model = train_td3(learning_rate=learnrate, 
                        gamma=gamma, 
                        batch_size=batch,
                        train_steps=steps)

