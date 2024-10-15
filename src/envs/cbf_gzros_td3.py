#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

class gymros:                                       # Class to create a queue for low frequency topics
    def __init__(self):
        self.sub_topic = '/response'                    # Topic name
        self.sub_msg_type = Float32MultiArray           # Message type
        self.queue = []                                 # Queue to store messages
        self.sub = rospy.Subscriber(self.sub_topic,     # Subscriber to the topic
                                    self.sub_msg_type,
                                    self.sub_callback,
                                    queue_size=10)
        self.pub = rospy.Publisher('/request', Float32MultiArray, queue_size=10)    # Publish request to gazebo simulation [cbf value , obstacle radius]

    def sub_callback(self, msg):                            # Callback function to append messages to the queue
        self.queue.append(msg.data)                     # Append the message to the queue

    def pop(self):                                      # Function to pop the first element from the queue
        return self.queue.pop(0)    

    def is_empty(self):                                 # Function to check if the queue is empty
        return len(self.queue) == 0

    def send_test_request(self, test_scenario):
        msg = Float32MultiArray()
        msg.data = test_scenario
        self.pub.publish(msg)

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # self.action_space = spaces.Box(low=0.001, high=1.50, shape=(1,), dtype=np.float32)
        
        self.cbf_max = 0.7000
        self.cbf_min = 0.002

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)    # normalised action space
        
        # self.observation_space = spaces.Box(low=0.1, high=10.0, shape=(1,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # normalised observation space
        
        self.obstacles = np.arange(0.2,5.1,0.2) # Obstacle radius from 0.2 to 5.0
        self.observation_space = spaces.Discrete(len(self.obstacles)) # Discrete observation space
        self.maxActPerObs = np.ones(len(self.obstacles)) * self.cbf_max     # initialise max action per observation array, max action will be replaced when negative reward is returned for the observation

        self.observation = None
        self.a = None
        self.r = None
        self.gymros = gymros()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = self.observation_space.sample()
        info = {}
        return self.observation, info

    def step(self, action):
        self.a = action
        print(f"[gym-step] New step(episode) to test CBF value: {self.a} with obstacle radius: {self.observation}")
        test_action = self.getTestAction()
        test_observation = self.getTestObservation()

        if test_action > self.maxActPerObs[self.observation]:   # check if action is greater than max action for the observation
            rospy.sleep(0.1)
            terminated = True                                   # Episode is complete
            truncated = True                                    # Episode should not count towards the learning (if true)
            reward = 0                                          # set reward to 0 if episode is truncated (shouldnt count towards learning)
            print(f"[gym-step] Action {test_action} is greater than max action {self.maxActPerObs[self.observation]} for obstacle radius {test_observation}m\n >> Skipping and Truncating This Test!") # info
        else:
            readyfortest = False
            while not readyfortest:
                rospy.sleep(0.1)
                readyfortest = not self.gymros.is_empty()       # if no messages in the queue, it is not ready
                if readyfortest:                                # when a message is in queue
                    print(f"[gym-step] Ready for test")             # debug
                    rsp = self.gymros.pop()                         # pop next message from queue
                    if rsp[2] >= 0:                                 # check the cbf_gamma value, will be -1.0 for new test request
                        readyfortest = False                        # if it is positive, sequence error and so not ready
                        print(f"[gym-step] Possible out of sequence message, expected new test vector, got: {rsp}")
            print(f"[gym-step] Gazebo response: {rsp}")         # ready for new test at this point
            test_scenario = [test_action, test_observation]          # create the test vector
            self.gymros.send_test_request(test_scenario)        # send the test vector to gazebo
            reward = None                                       # get ready to wait for reward response
            while reward == None:                               # until a reward is returned
                rospy.sleep(0.1)                                    # wait a bit
                if not self.gymros.is_empty():                      # if there is a message in the queue
                    response_from_gz = self.gymros.pop()                # get the response message
                    if response_from_gz[2] > 0:                     # check if the cbf_value returned is positive
                        reward = float(response_from_gz[0])             # if so the reward is first element of response
            print(f"[gym-step] Reward Value for CBF value {self.a} with obstacle radius {self.observation}m is : {reward}") # info
            terminated = True                                               # Episode is complete
            truncated = False                                               # Episode should not count towards the learning (if true)
            self.updateMaxActPerObs(reward,test_action, test_observation)   # update the max action for the observation
            if reward < -2:          # check for error run condition
                truncated = True        # if reward is < -1 episode is truncated
                reward = 0              # set reward to 0 if episode is truncated (shouldnt count towards learning)
        info = {}
        return self.observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Reward Value for CBF value {self.a} with obstacle radius {self.observation}m is : {self.r}")

    def getTestAction(self):
        # returns the actual action value from the normalised action value (must be updated in self.a before calling this)
        return self.cbf_min + (float(self.a) + 1.0) * (self.cbf_max - self.cbf_min) / 2.0
    
    def getTestObservation(self):
        # returns the test obstacle radius from the discrete observation space value
        return self.obstacles[self.observation]
    
    def updateMaxActPerObs(self, reward,test_acion,test_obs):
        if reward < 0:
            if test_acion < self.maxActPerObs[self.observation]:
                self.maxActPerObs[self.observation] = test_acion
                print(f"[gym-New Max CBF] Updated max CBF for obstacle {test_obs} to {test_acion}")        
        return

        # # convert normalised action to actual action
        # act_min = 0.005
        # act_max = 0.7000
        # test_action = act_min + (float(action) + 1.0) * (act_max - act_min) / 2.0
        # # convert normalised observation, which is sampled from observation space on reset to actual observation  (obstacle radius)
        # orad_min = 0.1
        # orad_max = 10.0
        # test_observation = orad_min + (float(self.observation) + 1.0) * (orad_max - orad_min) / 2.0