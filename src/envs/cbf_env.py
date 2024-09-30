import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.special import erf

def nonlinear_m(y):
    return 0.05*(y-5)**2    # define a non linear relationship between cbf_val distribution mean (m) and obstacle size (y)

# Skew-normal distribution
def skew_normal_pdf(x, y):
    sigma = 0.2
    m = nonlinear_m(y)
    alpha = -0.1 # 0 = no skew <1 = left skew, >1 = right skew
    # Standard normal PDF
    phi = np.exp(-0.5 * ((x - m) ** 2) / (sigma ** 2))
    # Standard normal CDF for the skew part
    Phi = 0.5 * (1 + erf(alpha * (x - m) / (sigma * np.sqrt(2))))
    # Combine to get the skew-normal PDF
    #return 2 * phi * Phi   # no negative values
    return 3 * phi * Phi - 0.5  # returns negative values for bad actions
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        
        self.action_space = spaces.Box(low=0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation = None  # Initialize observation attribute
        self.a = None
        self.r = None   
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = np.array([np.random.uniform(0, 3.0)], dtype=np.float32)
        info = {}
        return self.observation, info
    
    def step(self, action):
        self.a = action
        self.observation = np.array([np.random.uniform(0, 3.0)], dtype=np.float32)
        reward = self.custom_reward_function(self.observation, action)
        
        terminated = True  # Modify according to your specific termination condition
        truncated = False   # Modify according to any time-based or other truncation conditions
        info = {}
        
        return self.observation, reward, terminated, truncated, info
    
    def custom_reward_function(self, observation, action):
        self.r = skew_normal_pdf(action[0],observation[0])
        return self.r 
    
    def render(self, mode='human'):
        print(f"Reward Value for CBF value {self.a} with obstacle radius {self.observation}m is : {self.r}")