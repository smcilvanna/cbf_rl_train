#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from stable_baselines3 import SAC, TD3, DQN
import tkinter as tk
from tkinter import filedialog

class topicQueue:                                       # Class to create a queue for low frequency topics
    def __init__(self, topic, msg_type):
        self.topic = topic                              # Topic name
        self.msg_type = msg_type                        # Message type
        self.queue = []                                 # Queue to store messages
        self.sub = rospy.Subscriber(topic,              # Subscriber to the topic
                                    msg_type, 
                                    self.callback, 
                                    queue_size=10)  
    
    def callback(self, msg):                            # Callback function to append messages to the queue
        self.queue.append(msg.data)                     # Append the message to the queue
    
    def pop(self):                                      # Function to pop the first element from the queue
        return self.queue.pop(0)    
    
    def is_empty(self):                                 # Function to check if the queue is empty
        return len(self.queue) == 0

def select_load_model_file():  # Use to select file for processing
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])  # Open file dialog and get the file path
    
    fn = file_path.split('/')[-1]   # split out file_name.ext only
    if "td3" in fn:                 # determine what type of model is being loaded
        model_type = 1
    elif "sac" in fn:
        model_type = 2
    elif "dqn" in fn:
        model_type = 3
    else:
        model_type = None
        while model_type not in [1,2,3]:                                                # file names should include model type, but manual entry incase they dont
            model_type = input("What Type of Model is this?  1:TD3, 2:SAC, 3:DQN")      # Ask the user for the model type
            model_type = int(model_type)                                                # Convert the input to an integer
    
    if model_type == 1:                   # If the model is TD3
        model = TD3.load(file_path)              # Load the trained model
        mtxt = "TD3"
    elif model_type == 2:                 # If the model is SAC
        model = SAC.load(file_path)              # Load the trained model
        mtxt = "SAC"
    elif model_type == 3:                 # If the model is DQN
        model = DQN.load(file_path)              # Load the trained model
        mtxt = "DQN"
    
    return model, model_type, mtxt

def get_test_set():         # Get the test set from the trainer
    cbf_gammas = generate_test_set(start=0, step=0.1, stop=1, iterations=10)
    cbf_gammas = cbf_gammas[136:]
    obs_radii = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])


    grid1, grid2 = np.meshgrid(cbf_gammas, obs_radii)
    combinations = np.column_stack([grid1.ravel(), grid2.ravel()])
    # combinations = np.row_stack([combinations, combinations, combinations])
    # print("[TRAINER] Test Array Shape : ", combinations.shape) 
    return combinations

def generate_test_set(start=0, step=0.1, stop=1, iterations=5):
    all_numbers = []
    first_start = start
    first_stop = stop
    current_start = start
    current_step = step
    current_stop = stop
    
    for _ in range(iterations):
        # Generate the current set of numbers
        current_numbers = np.arange(current_start, current_stop + current_step, current_step)
        
        # Ensure the stop value is included
        if current_numbers[-1] > current_stop:
            current_numbers = current_numbers[:-1]
        
        # Convert to list and remove duplicates
        current_numbers_set = set(current_numbers.round(5))
        
        # Remove any numbers that are already in the all_numbers list
        current_numbers_set.difference_update(all_numbers)
        
        # Append unique numbers to the all_numbers list
        all_numbers.extend([num for num in current_numbers if num in current_numbers_set])
        
        # Update for the next iteration
        current_step /= 2
        current_start = first_start + current_step
        current_stop  = first_stop - current_step
    
    return np.array(all_numbers[1:])

def setup_scenario(test):                          # Get obstacle and target positions for the test scenario 
    husky_radius = 0.55                                 # Husky robot radius
    orad = test[1]                                      # Get the CBF parameter and obstacle radius from the trainer for next episode
    approach_sep = 10                                   # Approach separation from the obstacle
    target_sep = 5                                      # Target separation from the obstacle
    obstacle_x = husky_radius + orad + approach_sep     # Calculate the obstacle x position
    target_x =   obstacle_x + orad + target_sep         # Calculate the target x position
    return obstacle_x, target_x

def continious_normal_action_to_cbf(value,new_min=0.005,new_max=0.70):
    old_min = -1.0 ; old_max = 1.0                          # Normalised actions produced from model
    scale = (new_max - new_min) / (old_max - old_min)       # Calculate the scaling factor
    cbf_value = float(new_min + (value - old_min) * scale) # Apply the scaling to get real CBF value for NMPC-CBF
    return cbf_value

def discrete_action_to_cbf(discrete_value,new_min=0.002,new_max=0.70,step=0.002):
    action_array = np.arange(new_min, new_max, step)
    return action_array[discrete_value]

def obstacle_radius_to_discrete_observation(obstacle_radius, min_obs=0.2,max_obs=5.0,step_obs=0.2):
    obstacle_array = np.arange(min_obs, max_obs+step_obs, step_obs).reshape(-1,1)
    discrete_observation = np.where(np.isclose(obstacle_array, obstacle_radius))[0][0]
    return discrete_observation

def ask_model(model, model_type, obstacle_radius):
    observation = obstacle_radius_to_discrete_observation(obstacle_radius)
    model_action = model.predict(observation,deterministic=True)[0]
    if model_type in [1,2]:
        cbf_action = continious_normal_action_to_cbf(model_action)
    else:
        cbf_action = discrete_action_to_cbf(model_action)    
    return cbf_action

def trainer_node():                                                         # Main function to run NMPC
    manual_entry = True                                                         # Enable manual entry of test scenarios                             
    test_set= get_test_set()                                                    # Get the test set          
    # test_set = generate_test_set(start=0, step=0.1, stop=10, iterations=15)      # Generate the test set
    print(f"\n\n########[TRAINER]########\nStarting {test_set.shape[0]} tests\n#########################\n\n")                                  # Print the test set
    rospy.init_node("dummy_trainer", anonymous=True)                            # Init ROS node
    pub_request= rospy.Publisher('/request', Float32MultiArray, queue_size=10)  # Publisher for request
    response = topicQueue('/response', Float32MultiArray)                       # Subscriber queue for response
    r = rospy.Rate(10)                                                          # Rate of the node                
    # obstacle = GazeboCylinderManager()

    model, model_type, mtxt = select_load_model_file()

    test_idx = 0                                                                # Initialize test index
    while not rospy.is_shutdown():
        while response.is_empty():              # wait for response from trainer
            rospy.sleep(0.1)                        # wait here until trainer responds
        rsp = response.pop()                    # pop the prompt from the queue
        if (rsp[0]==-1 and rsp[1]==-1 and rsp[2]==-1) :     # check if response is a prompt 
            if manual_entry:                                    # check if manual entry is enabled
                obs_radius = None                               # initialize obs_radius     

                while obs_radius is None:       # loop until obs_radius is entered correctly
                    obs_radius = input("And finally what obs_radius value to test? : ")
                    try:
                        obs_radius = float(obs_radius)
                    except:
                        print("Invalid input, please enter a float value")
                        obs_radius = None

                cbf_gamma = ask_model(model,model_type,obs_radius)
                print("[TRAINER] Model scenario Requested, sending :", [cbf_gamma, obs_radius])      # print the manual test scenario
                pub_request.publish(Float32MultiArray(data=[cbf_gamma, obs_radius]))                # publish the manual test scenario
            else:
                this_test = test_set[test_idx]                                                     # get the test scenario
                print("[TRAINER] Test scenario Requested, sending :", this_test)           # print the test scenario
                pub_request.publish(Float32MultiArray(data=this_test))                     # publish the test scenario
                
                # if test_idx != 0:                                               # on first run spawn the cylinder
                #     obstacle.delete_cylinder()                                  # delete the obstacle
                #     rospy.sleep(0.2)
                
                # target_x, obstacle_x = setup_scenario(this_test)                # get the obstacle and target positions
                # obstacle.spawn_cylinder(obstacle_x, 0, this_test[1])            # spawn the obstacle
                
                test_idx += 1                                                                       # increment the test index                         
        else:
            print("[TRAINER] Results from test : ", rsp)                                # print the results from the test
            if test_idx >= len(test_set):                                           # check if all test scenarios are done
                print("[TRAINER] Test scenario Requested, sending :", [-1.0, -1.0])      # print the end test scenario msg
                pub_request.publish(Float32MultiArray(data=[-1.0, -1.0]))                # publish the end test scenario msg
                break                                                                    # break the loop                   
        r.sleep()                                                                   # sleep until rate is met

if __name__ == '__main__':                                  # Main function to run the node
    try:                                                        # Try to run the node
        trainer_node()                                              # Run the node
    except rospy.ROSInterruptException:                         # Catch the exception
        pass                                                        # Pass the exception
