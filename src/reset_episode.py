#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
import numpy as np

def wraptopi(x):   # used to wrap angle errors to interval [-pi pi]
    pi = np.pi  
    x = x - np.floor(x/(2*pi)) *2*pi
    if x > pi:
        x = x - 2*pi
    return x

def getyaw(odom):        # Convert quaternion to euler
    orientation = odom.orientation
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    roll, pitch, yaw = euler_from_quaternion(q)
    wrap_yaw = wraptopi(yaw)
    return round(wrap_yaw,2)

def check_reset(tols=[0.02, 0.02, 0.2, 0.005]): # check if reset is complete
    rospy.sleep(0.3)                                # sleep for a short time to allow husky to settle
    xtol = tols[0]                                  # x position tolerance
    ytol = tols[1]                                  # y position tolerance
    ztol = tols[2]                                  # z position tolerance
    wtol = tols[3]                                  # yaw tolerance
    check_gazebo = rospy.wait_for_message('/gazebo/model_states', ModelStates)  # get model states
    hidx = check_gazebo.name.index('husky')         # get index of husky in model states
    husky_pose = check_gazebo.pose[hidx]            # get pose of husky
    x = abs(husky_pose.position.x) < xtol           # check if x position is within tolerance
    y = abs(husky_pose.position.y) < ytol           # check if y position is within tolerance
    # z = abs(husky_pose.position.z) < ztol         # check if z position is within tolerance
    yaw = abs(getyaw(husky_pose)) < wtol            # check if yaw is within tolerance
    check_ekf = rospy.wait_for_message('/odometry/filtered', Odometry)                          # get current ekf odometry
    ex = check_ekf.pose.pose.position.x                                                         # get x position from ekf odometry                                    
    ey = check_ekf.pose.pose.position.y                                                         # get y position from ekf odometry
    print(f"\n\n#######################################\n[RESET]: EKF New X: {ex}  Y: {ey}")    # debug print ekf position
    ekfx = abs(ex) < xtol                                                                       # check if x position is within tolerance
    ekfy = abs(ey) < ytol                                                                       # check if y position is within tolerance
    ekf_zero = ekfx and ekfy                                                                    # check if both ekf x and y are within tolerance
    gz_zero = x and y and yaw                                                                   # check if gazebo x, y and yaw are within tolerance
    reset_good = ekf_zero and gz_zero                                                           # check if both ekf and gazebo are both good
    print(f"Gazebo is reset : {gz_zero}  | EKF is reset : {gz_zero}    | "
          f"Reset Good : {reset_good}\n#######################################\n\n")            # debug print reset status
    return reset_good                                                                           # return True if all conditions are met

def init_poses():
    # Pose that will be applied on reset (for husky ekf node that provides odom) 
    rpose = PoseWithCovarianceStamped()
    rpose.header.frame_id = 'odom'
    rpose.pose.pose.position.x = 0
    rpose.pose.pose.position.y = 0
    rpose.pose.pose.position.z = 0
    rpose.pose.pose.orientation.w = 1
    rpose.pose.pose.orientation.x = 0
    rpose.pose.pose.orientation.y = 0
    rpose.pose.pose.orientation.z = 0

    # Pose that will be applied on reset (for gazebo model state)
    rstate = ModelState()
    rstate.model_name = 'husky'
    rstate.pose.position.x = 0
    rstate.pose.position.y = 0
    rstate.pose.position.z = 0.135  # set this so reset position is not below ground, but not so high as to bounce husky
    rstate.pose.orientation.w = 1
    rstate.pose.orientation.x = 0
    rstate.pose.orientation.y = 0
    rstate.pose.orientation.z = 0

    return rpose, rstate

if __name__ == "__main__":
    unpause         = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
    pause           = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
    reset_proxy     = rospy.ServiceProxy("/gazebo/reset_world", Empty)
    set_state       = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10  )
    set_pose        = rospy.Publisher("/set_pose", PoseWithCovarianceStamped, queue_size=10  )
    # set_ekf         = rospy.Publisher("/ekf_odom/set_pose", PoseWithCovarianceStamped, queue_size=1  )  # for real husky
    rpose, rstate   = init_poses()                          # get reset poses
    rospy.init_node("reset_episode", anonymous=True)        # initialize node
    rcnt = 0                                                # reset count
    done = False                                            # flag to check if reset is complete
    while not done:                                         # loop until reset is complete

        pause()                                                 # pause physics                      
        rospy.wait_for_service("/gazebo/reset_world")           # reset gazebo world      
        try:
            reset_proxy()                                       # call reset world service
        except rospy.ServiceException as e:                     # catch exception if service call fails
            rospy.logwarn("/gazebo/reset_simulation service call failed")
        rcnt += 1                                               # increment reset count
        set_state.publish(rstate)                               # set husky pose in gazebo
        set_pose.publish(rpose)                                 # set husky ekf pose
    
        rospy.wait_for_service("/gazebo/unpause_physics")       # unpause physics
        try:
            unpause()                                           # call unpause physics service 
        except (rospy.ServiceException) as e:                   # catch exception if service call fails
            rospy.logwarn("/gazebo/unpause_physics service call failed")
        rospy.sleep(0.1)                                       # sleep for a short time to allow husky to settle
        done = check_reset([0.02, 0.02, 0.2, 0.01])             # check if reset is complete [x, y, z, yaw] (tolerances)
        
    print("[RESET] Reset complete after {} attempts".format(rcnt))     # print number of attempts to reset
