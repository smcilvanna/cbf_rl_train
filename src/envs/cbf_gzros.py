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
        self.action_space = spaces.Box(low=0, high=1.50, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation = None #np.array([0.5], dtype=np.float32)  # Initialize observation attribute
        self.a = None
        self.r = None
        self.gymros = gymros()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options == None:# if no options are provided, set the obstacle radius to a random value between 0 and 3
            rand_obs = np.array([np.random.uniform(0, 3.0)], dtype=np.float32)
            self.observation = rand_obs.round(1)
        else:
            self.observation = np.array([options['orad']], dtype=np.float32)
        info = {}
        return self.observation, info
    
    def step(self, action):
        self.a = action
        print(f"[gym-step] New step(episode) to test CBF value: {self.a} with obstacle radius: {self.observation}")
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
        test_scenario = [self.a, self.observation]          # create the test vector
        self.gymros.send_test_request(test_scenario)        # send the test vector to gazebo
        reward = None                                       # get ready to wait for reward response
        while reward == None:                               # until a reward is returned
            rospy.sleep(0.1)                                    # wait a bit
            if not self.gymros.is_empty():                      # if there is a message in the queue
                response_from_gz = self.gymros.pop()                # get the response message
                print(response_from_gz)                             # debug
                print(type(response_from_gz))                       # debug
                print(len(response_from_gz))                        # debug
                if response_from_gz[2] > 0:                     # check if the cbf_value returned is positive
                    reward = float(response_from_gz[0])             # if so the reward is first element of response
        print(f"[gym-step] Reward Value for CBF value {self.a} with obstacle radius {self.observation}m is : {reward}") # info
        terminated = True                                   # Episode is complete
        truncated = False                                   # Episode should not count towards the learning (if true)
        info = {}
        
        return self.observation, reward, terminated, truncated, info
    
    # def custom_reward_function(self, action):
    #     self.r = skew_normal_pdf(action[0],self.observation[0])
    #     return self.r 
    
    def render(self, mode='human'):
        print(f"Reward Value for CBF value {self.a} with obstacle radius {self.observation}m is : {self.r}")




