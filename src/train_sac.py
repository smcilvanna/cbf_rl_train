#!/usr/bin/env python3
import gymnasium as gym
import rospy
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import numpy as np

################# Register Custom Environment ###############################
from gymnasium.envs.registration import register
register(
    id='cbf-train-gzros-td3',  # Unique identifier for the environment
    entry_point='envs.cbf_gzros_td3:CustomEnv',  # Corrected path
    max_episode_steps=1,  # Maximum number of steps per episode
)
#############################################################################








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

def train_sac(learning_rate=1e-3, gamma=0.00, batch_size=32, train_steps=5000):    
    env = gym.make('cbf-train-gzros-td3')          # Create the environment
    n_acions = env.action_space.shape[-1]                               # Number of actions (for action noise parameter)
    tb_path = "/home/user/husky/tensorboard/sac"                        # Path to the directory where TensorBoard logs will be saved
    action_noise = NormalActionNoise(mean=np.zeros(n_acions),
                                      sigma=0.1 * np.ones(n_acions))    # Action noise to explore the environment, increase sigma for more noise/exploration
    
    model = SAC(                        # Define the SAC model
        "MlpPolicy",                                # Policy type (multi-layer perceptron)
        env,                                        # the custom environment
        verbose=                1,                  # Verbosity mode
        learning_rate=          learning_rate,      # Learning rate passed as an argument
        gamma=                  gamma,              # Discount factor passed as an argument
        batch_size=             batch_size,         # Batch size for training
        action_noise=           action_noise,       # Action noise to encourage exploration
        buffer_size=            5000,               # Replay buffer size
        learning_starts=        1,                  # Number of steps before training starts
        tau=                    0.005,              # Target network update coefficient
        policy_kwargs=          dict(net_arch=[64, 64])  # Policy network architecture
    )

    checkpoint_callback = CustomCheckpointCallback(       # Callback to save the model and replay buffer every 100 steps/episodes
        save_freq=100, 
        save_path='/home/user/husky/sac_models/',
        name_prefix=f"sac_run{run_id}_chkpt"
    )

    # Train the model
    model.learn(total_timesteps=train_steps,
                callback=checkpoint_callback)       # Training with checkpoint callback
    
    n = model._n_updates
    # Save the model and replay buffer
    model.save(f"/home/user/husky/sac_models/final/sac_run{run_id}_model_{n}")          
    model.save_replay_buffer(f"/home/user/husky/sac_models/final/sac_run{run_id}_replay_buffer_{n}") 
    max_action_array = env.maxActPerObs
    np.save(f"/home/user/husky/sac_models/final/sac_run{run_id}_{n}_max_action.npy",max_action_array)
    return model

if __name__ == "__main__":
    run_id = 1
    rospy.init_node('gymnode', anonymous=True)      # Initialize the node
    steps = 20000
    model = train_sac(train_steps=steps)
