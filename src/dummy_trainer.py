#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

def get_test_set():         # Get the test set from the trainer
    cbf_gammas = np.linspace(0.1, 2.0, 20)
    obs_radii = np.arange(0.5, 5.0, 0.5)
    grid1, grid2 = np.meshgrid(cbf_gammas, obs_radii)
    combinations = np.column_stack([grid1.ravel(), grid2.ravel()])
    return combinations

response = []

def response_callback(msg):    # Callback function to receive the response from the trainer
    global response
    response.append(msg.data)


def trainer_node():            # Main function to run NMPC
    
    test_set= get_test_set()
    rospy.init_node("dummy_trainer", anonymous=True)                  # Init ROS node
    pub_request= rospy.Publisher('/request', Float32MultiArray, queue_size=10)
    sub_response= rospy.Subscriber('/response', Float32MultiArray, response_callback) # Subscribe to the response from the trainer
    r = rospy.Rate(100)
    test_idx = 0

    while not rospy.is_shutdown():
        
        # request_prompt = rospy.wait_for_message('/response', Float32MultiArray) # wait for prompt from trainer
        while len(response) == 0:
            rospy.sleep(0.1)

        rsp = response.pop(0)

        if rsp[0] < 0:                                       # check if prompt is for test set
            print("[TRAINER] Test scenario Requested, sending :", test_set[test_idx])
            
            pub_request.publish(Float32MultiArray(data=test_set[test_idx]))
            test_idx += 1
        else:
            print("[TRAINER] Results from test : ", rsp)

        if test_idx == len(test_set):
            break

        r.sleep()


if __name__ == '__main__':    
    
    try:
        trainer_node()
    except rospy.ROSInterruptException:
        pass
