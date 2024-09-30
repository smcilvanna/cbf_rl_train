import gymnasium as gym
import register_env  # need this so module registers the custom environment!
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from itertools import product
import time
import numpy as np
import os

def train_td3(learning_rate, gamma, batch_size, train_steps=20000):    
    env = gym.make('cbf-train-gzros')          # Create the environment
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))  # Add minimum noise to the action space
    model = TD3(                        # Define the TD3 model
        "MlpPolicy",                                # Policy type (multi-layer perceptron)
        env,                                        # the custom environment
        verbose=                1,                  # Verbosity mode
        learning_rate=          learning_rate,      # Learning rate passed as an argument
        gamma=                  gamma,              # Discount factor passed as an argument
        batch_size=             batch_size,         # Batch size for training
        buffer_size=            1000,               # Replay buffer size
        learning_starts=        10,                 # Number of steps before training starts
        policy_delay=           1,                  # Policy update delay 1=update immediately (no delay in reward in this scenario)
        action_noise=           action_noise,       # Action noise (minimal noise)
        target_policy_noise=    0.0,
        policy_kwargs=          dict(net_arch=[64, 64]),  # Policy network architecture
        #tensorboard_log="./td3_tensorboard/"       # Path to the directory where TensorBoard logs will be saved, uncomment for logging
        tau=            0.05                        # Target network update coefficient, slightly larger for deterministic environment
    )

    # Train the model
    #model.learn(total_timesteps=train_steps, log_interval=10)  # with tensorboard logging
    model.learn(total_timesteps=train_steps)                    # without tensorboard logging
    return model

# def evaluate_trained_model(model,i,savecsv=False, val_steps=100):
#     # Evaluate the trained model
#     env = gym.make('cbf-value-env-v2')  # Create the environment
#     obs, info = env.reset()
#     sum_reward = 0
#     min_reward = 999

#     for episode in range(val_steps):  # Run for N episodes
#         done = False
#         while not done:
#             action, _states = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             env.a = action
#             env.r = reward
#             env.render()  # Optionally render the environment
#             # Check if the episode is finished
#             done = terminated or truncated
#             sum_reward += reward
#             min_reward = min(min_reward, reward)
#         obs, info = env.reset()
#         ave_reward = sum_reward / val_steps    
        
#         results[i,0] = test_set[i][0]                           # record results
#         results[i,1] = test_set[i][1]
#         results[i,2] = test_set[i][2]
#         results[i,3] = ave_reward
#         results[i,4] = min_reward
#         results[i,5] = steps
#         results[i,6] = int(i+1)        
        
#         if savecsv:       # set true to save to file, false for print only
#             outfile = str(i+1) + '_v2.1_results_50k_.csv'     # name of output file for results
#             np.savetxt(outfile, results, delimiter=",")             # save results to file after each test
#         else:
#             print("Trained TD3 model with ", steps, " steps(episodes).")
#             print("Average reward: ", ave_reward, " Min reward: ", min_reward)

if __name__ == "__main__":

    learnrate = 1e-3
    gamma     = 0.00
    batch     = 32

    steps = 100
    model = train_td3(learning_rate=learnrate, 
                        gamma=gamma, 
                        batch_size=batch,
                        train_steps=steps)
    # mdlsvnm = "./" + str(i+1) + "_td3_model_" + str(steps) + "steps"
    # model.save(mdlsvnm)

