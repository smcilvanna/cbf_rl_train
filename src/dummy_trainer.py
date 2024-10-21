#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3

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

# class GazeboCylinderManager:
#     def __init__(self):
#         # rospy.init_node('cylinder_manager', anonymous=True)
#         self.spawn_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
#         self.delete_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
#         self.model_name = "unit_cylinder"
#         self.model_path = "model://cylinder"  # Reference to Gazebo's built-in cylinder
#     def spawn_cylinder(self, x, y, radius):
#         scale = Vector3(radius, radius, 3)  # Set scale for radius and height
        
#         # model_xml = f"""
#         #             <robot name="{self.model_name}">
#         #             <link name="body">
#         #                 <visual>
#         #                 <geometry>
#         #                     <cylinder radius="{radius}" length="{radius}"/>
#         #                 </geometry>
#         #                 </visual>
#         #                 <collision>
#         #                 <geometry>
#         #                     <cylinder radius="{radius}" length="{radius}"/>
#         #                 </geometry>
#         #                 </collision>
#         #             </link>
#         #             </robot>
#         #             """

#         pose = Pose(Point(x, y, 1), Quaternion(0, 0, 0, 1))
#         try:
#             self.spawn_service(self.model_name, self.model_path, "", pose, "world")
#             rospy.loginfo("Cylinder spawned successfully")
#         except rospy.ServiceException as e:
#             rospy.logerr(f"Failed to spawn cylinder: {e}")

#     def delete_cylinder(self):
#         try:
#             self.delete_service(self.model_name)
#             rospy.loginfo("Cylinder deleted successfully")
#         except rospy.ServiceException as e:
#             rospy.logerr(f"Failed to delete cylinder: {e}")

#######################################################################################################################

def get_test_set():         # Get the test set from the trainer
    # cbf_gammas = np.linspace(0.1, 2.0, 20)
    # obs_radii = np.arange(0.5, 5.0, 0.5)

    # cbf_gammas = np.ones(5)*0.1
    # # obs_radii = np.ones(2)*1.5
    # obs_radii = np.array([3.5, 4.5, 5.5])

    # cbf_gammas = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    # obs_radii = np.array([1.0, 5.0])

    # cbf_gammas = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    # obs_radii = np.array([1.0])

    # cbf_gammas = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0,1.5])
    cbf_gammas = generate_test_set(start=0, step=0.1, stop=10, iterations=3)
    obs_radii = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])


    grid1, grid2 = np.meshgrid(cbf_gammas, obs_radii)
    combinations = np.column_stack([grid1.ravel(), grid2.ravel()])
    # combinations = np.row_stack([combinations, combinations, combinations])
    # print("[TRAINER] Test Array Shape : ", combinations.shape) 
    return combinations

def generate_test_set(start=0, step=0.1, stop=10, iterations=3):
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

def trainer_node():                                                         # Main function to run NMPC
    manual_entry = False                                                         # Enable manual entry of test scenarios                             
    test_set= get_test_set()                                                    # Get the test set          
    # test_set = generate_test_set(start=0, step=0.1, stop=10, iterations=15)      # Generate the test set
    print(f"\n\n########[TRAINER]########\nStarting {test_set.shape[0]} tests\n#########################\n\n")                                  # Print the test set
    rospy.init_node("dummy_trainer", anonymous=True)                            # Init ROS node
    pub_request= rospy.Publisher('/request', Float32MultiArray, queue_size=10)  # Publisher for request
    response = topicQueue('/response', Float32MultiArray)                       # Subscriber queue for response
    r = rospy.Rate(10)                                                          # Rate of the node                
    # obstacle = GazeboCylinderManager()
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
