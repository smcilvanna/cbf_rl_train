#!/usr/bin/env python3
from gymnasium.envs.registration import register

# Import your environment class
from envs.cbf_gzros import CustomEnv  # Adjust the import path accordingly

# # Register the custom environment with Gymnasium
# register(
#     id='cbf-value-env-v1',  # Unique identifier for the environment
#     entry_point='envs.cbf_env:CustomEnv',  # Corrected path
#     max_episode_steps=1,  # Maximum number of steps per episode
# )

# register(
#     id='cbf-value-env-v2',  # Unique identifier for the environment
#     entry_point='envs.cbf_obs_env:CustomEnv',  # Corrected path
#     max_episode_steps=1,  # Maximum number of steps per episode
# )

register(
    id='cbf-train-gzros',  # Unique identifier for the environment
    entry_point='envs.cbf_gzros:CustomEnv',  # Corrected path
    max_episode_steps=1,  # Maximum number of steps per episode
)