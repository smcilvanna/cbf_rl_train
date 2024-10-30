#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import os
import rospy

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

def train_td3(train_steps=100):    
    env = gym.make('cbf-train-gzros-td3')                               # Create the environment
    n_acions = env.action_space.shape[-1]                               # Number of actions (for action noise parameter)
    tb_path = "/home/user/husky/tensorboard/td3"                        # Path to the directory where TensorBoard logs will be saved
    action_noise = NormalActionNoise(mean=np.zeros(n_acions),
                                      sigma=0.2 * np.ones(n_acions))    # Action noise to explore the environment, increase sigma for more noise/exploration
    model = TD3(                                          # Define the TD3 model
        "MlpPolicy",                                        # Policy type (multi-layer perceptron)
        env,                                                # the custom environment
        learning_rate=          5e-4,              # Learning rate passed as an argument
        gamma=                  0.00,                      # Discount factor passed as an argument
        batch_size=             32,                 # Batch size for training
        action_noise=           action_noise,               # Action noise (minimal noise)
        policy_kwargs=          dict(net_arch=[64, 64]),    # Policy network architecture ######## COULD CONSIDER ADDING : learning_rate_actor=1e-3, learning_rate_critic=1e-4 for separate learning rates 
        tensorboard_log=        tb_path,                    # Tensorboard path
        verbose=                1,                          # Verbosity mode
        buffer_size=            10000,                       # Replay buffer size
        learning_starts=        1,                          # Number of steps before training starts
        policy_delay=           2,                          # Policy update delay 2 : actor (policy) network is updated once for every two updates of the critic (Q) network, allows the critic to better estimate the value function before updating the policy
        target_policy_noise=    0.2,                        # Adds noise to target actions which promotes exploration during updates, helping avoid local optima
        tau=                    0.05                        # Target network update coefficient, slightly larger for deterministic environment
    )

    checkpoint_callback = CustomCheckpointCallback(       # Custom callback to save the model and replay buffer during training in case of crash
        save_freq=100, 
        save_path='/home/user/husky/td3_models/',
        name_prefix=f"td3_run{run_id}_chkpt")

    # Train the model
    model.learn(total_timesteps=train_steps,
                callback=checkpoint_callback,
                log_interval=1)                             # Train TD3 model
    
    n = model._n_updates                                                                                 # Number of training runs for file tag
    model.save(f"/home/user/husky/td3_models/final/td3_run{run_id}_model_{n}")                           # Save the model
    model.save_replay_buffer(f"/home/user/husky/td3_models/final/td3_run{run_id}_replay_buffer_{n}")     # Save the replay buffer
    max_action_array = env.maxActPerObs
    np.save(f"/home/user/husky/td3_models/final/td3_run{run_id}_{n}_max_action.npy",max_action_array)
    return model

if __name__ == "__main__":
    run_id = 7                                      # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Run ID for file tag - CHANGE EVERY TRAINING SESSION !!!!!!!!!
    rospy.init_node('gymnode', anonymous=True)      # Initialize the node
    steps = 2000                                    # Number of training steps
    model = train_td3(train_steps=steps)            # Train the TD3 model
    print(">>>>>>>>>>>>>===============<<<<<<<<<<<<<<\n           TRAINING COMPLETE!             \n>>>>>>>>>>>>>===============<<<<<<<<<<<<<<")






##########################################################################################################################################################################################
# Run 6

# def train_td3(learning_rate= 1e-3, gamma= 0.00, batch_size= 32, train_steps=100):    
#     env = gym.make('cbf-train-gzros-td3')                               # Create the environment
#     n_acions = env.action_space.shape[-1]                               # Number of actions (for action noise parameter)
#     tb_path = "/home/user/husky/tensorboard/td3"                        # Path to the directory where TensorBoard logs will be saved
#     action_noise = NormalActionNoise(mean=np.zeros(n_acions),
#                                       sigma=0.2 * np.ones(n_acions))    # Action noise to explore the environment, increase sigma for more noise/exploration
#     model = TD3(                                          # Define the TD3 model
#         "MlpPolicy",                                        # Policy type (multi-layer perceptron)
#         env,                                                # the custom environment
#         learning_rate=          learning_rate,              # Learning rate passed as an argument
#         gamma=                  gamma,                      # Discount factor passed as an argument
#         batch_size=             batch_size,                 # Batch size for training
#         action_noise=           action_noise,               # Action noise (minimal noise)
#         policy_kwargs=          dict(net_arch=[64, 64]),    # Policy network architecture ######## COULD CONSIDER ADDING : learning_rate_actor=1e-3, learning_rate_critic=1e-4 for separate learning rates 
#         tensorboard_log=        tb_path,                    # Tensorboard path
#         verbose=                1,                          # Verbosity mode
#         buffer_size=            5000,                       # Replay buffer size
#         learning_starts=        1,                          # Number of steps before training starts
#         policy_delay=           2,                          # Policy update delay 2 : actor (policy) network is updated once for every two updates of the critic (Q) network, allows the critic to better estimate the value function before updating the policy
#         target_policy_noise=    0.2,                        # Adds noise to target actions which promotes exploration during updates, helping avoid local optima
#         tau=                    0.05                        # Target network update coefficient, slightly larger for deterministic environment
#     )

    ##########################################################################################################################################################################################