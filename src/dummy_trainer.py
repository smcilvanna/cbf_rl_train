#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

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

#######################################################################################################################

def get_test_set():         # Get the test set from the trainer
    # cbf_gammas = np.linspace(0.1, 2.0, 20)
    # obs_radii = np.arange(0.5, 5.0, 0.5)
    
    cbf_gammas = np.ones(3)*0.1
    # obs_radii = np.ones(2)*1.5
    obs_radii = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    grid1, grid2 = np.meshgrid(cbf_gammas, obs_radii)
    combinations = np.column_stack([grid1.ravel(), grid2.ravel()])
    
    combinations = np.row_stack([combinations, combinations, combinations])
    
    print("[TRAINER] Test Array Shape : ", combinations.shape) 
    
    
    return combinations

def trainer_node():                                                         # Main function to run NMPC
    manual_entry = True                                                         # Enable manual entry of test scenarios                             
    test_set= get_test_set()                                                    # Get the test set          
    rospy.init_node("dummy_trainer", anonymous=True)                            # Init ROS node
    pub_request= rospy.Publisher('/request', Float32MultiArray, queue_size=10)  # Publisher for request
    response = topicQueue('/response', Float32MultiArray)                       # Subscriber queue for response
    r = rospy.Rate(10)                                                          # Rate of the node                
    test_idx = 0                                                                # Initialize test index
    while not rospy.is_shutdown():
        while response.is_empty():              # wait for response from trainer
            rospy.sleep(0.1)                        # wait here until trainer responds
        rsp = response.pop()                    # pop the prompt from the queue
        if rsp[0] < 0:                          # check if response is a prompt 
            if manual_entry:                    # check if manual entry is enabled
                cbf_gamma = None                # initialize cbf_gamma
                obs_radius = None               # initialize obs_radius     
                while cbf_gamma is None:        # loop until cbf_gamma is entered correctly
                    cbf_gamma = input("New Test Scenario Requested, what cbf_gamma value to test? : ")
                    try:
                        cbf_gamma = float(cbf_gamma)
                    except:
                        print("Invalid input, please enter a float value")
                        cbf_gamma = None
                while obs_radius is None:       # loop until obs_radius is entered correctly
                    obs_radius = input("And finally what obs_radius value to test? : ")
                    try:
                        obs_radius = float(obs_radius)
                    except:
                        print("Invalid input, please enter a float value")
                        obs_radius = None
            
                print("[TRAINER] Test scenario Requested, sending :", [cbf_gamma, obs_radius])      # print the manual test scenario
                pub_request.publish(Float32MultiArray(data=[cbf_gamma, obs_radius]))                # publish the manual test scenario
            else:
                print("[TRAINER] Test scenario Requested, sending :", test_set[test_idx])           # print the test scenario
                pub_request.publish(Float32MultiArray(data=test_set[test_idx]))                     # publish the test scenario
                test_idx += 1                                                                       # increment the test index                         
        else:
            print("[TRAINER] Results from test : ", rsp)                                # print the results from the test
            if test_idx == len(test_set):                                           # check if all test scenarios are done
                print("[TRAINER] Test scenario Requested, sending :", [-1.0, -1.0])      # print the end test scenario msg
                pub_request.publish(Float32MultiArray(data=[-1.0, -1.0]))                # publish the end test scenario msg
                break                                                                    # break the loop                   
        r.sleep()                                                                   # sleep until rate is met

if __name__ == '__main__':                                  # Main function to run the node
    try:                                                        # Try to run the node
        trainer_node()                                              # Run the node
    except rospy.ROSInterruptException:                         # Catch the exception
        pass                                                        # Pass the exception
