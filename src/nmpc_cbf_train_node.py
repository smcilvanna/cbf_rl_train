#!/usr/bin/env python3

import numpy as np
import math
import time
import casadi as ca
import rospy
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Int16, Float32, Int8, Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion
import subprocess
import signal
import os
import atexit
import rosbag
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from process_rosbag import get_reward


# ###################################################################################################
# CLASS DEFINITIONS                                                                                           
# ###################################################################################################

class NMPC_CBF_Terminal:
    def __init__(self, init_pos, limitation, DT, N, W_q, W_r, W_v, cbf_gamma, SO):
        self.DT = DT            # time step
        self.N = N              # horizon length
        self.W_q = W_q          # Weight matrix for states
        self.W_r = W_r          # Weight matrix for controls
        self.W_v = W_v          # Weight matrix for Terminal state
        self.min_x = limitation[0] 
        self.max_x = limitation[1]
        self.min_y = limitation[2] 
        self.max_y = limitation[3]
        self.min_theta = limitation[4]
        self.max_theta = limitation[5]
        self.min_v = limitation[6]
        self.max_v = limitation[7]
        self.min_omega = limitation[8]
        self.max_omega = limitation[9]
        self.max_dv = 0.8
        self.max_domega = math.pi/7
        self.R_husky = 0.55             ########################################################################
        self.cbf_gamma = cbf_gamma
        self.SO = SO
        self.n_SO = len(self.SO[:, 0])
        # Initial value for state and control input
        self.next_states = np.ones((self.N+1, 3))*init_pos
        self.u0 = np.zeros((self.N, 2))
        self.setup_controller()
    
    def setup_controller(self):
        self.opt = ca.Opti()
        self.opt_states = self.opt.variable(self.N+1, 3)
        x = self.opt_states[:,0]
        y = self.opt_states[:,1]
        theta = self.opt_states[:,2]
        self.opt_controls = self.opt.variable(self.N, 2)
        v = self.opt_controls[:,0]
        omega = self.opt_controls[:,1]

        # dynamic mapping function
        f = lambda x_, u_: ca.vertcat(*[ca.cos(x_[2])*u_[0],  
                                        ca.sin(x_[2])*u_[0], 
                                        u_[1]])      
                                              
        # these parameters are the reference trajectories of the state and inputs
        self.u_ref = self.opt.parameter(self.N, 2)
        self.x_ref = self.opt.parameter(self.N+2, 3)

        # dynamic constraint
        self.opt.subject_to(self.opt_states[0,:] == self.x_ref[0,:])
        for i in range(self.N):            
            st = self.opt_states[i,:]
            ct = self.opt_controls[i,:]
            k1 = f(st,ct).T
            k2 = f(st + self.DT/2*k1, ct).T
            k3 = f(st + self.DT/2*k2, ct).T
            k4 = f(st + self.DT*k3, ct).T
            x_next = st + self.DT/6*(k1 + 2*k2 + 2*k3 + k4)
            self.opt.subject_to(self.opt_states[i+1,:] == x_next)

        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i,:] - self.x_ref[i+1,:]
            control_error_ = self.opt_controls[i,:] - self.u_ref[i,:]
            obj = obj + ca.mtimes([state_error_, self.W_q, state_error_.T]) + ca.mtimes([control_error_, self.W_r, control_error_.T])
        #state_error_N = self.opt_states[self.N,:] - self.x_ref[self.N+1,:]
        #obj = obj + ca.mtimes([state_error_N, self.W_v, state_error_N.T])    
        self.opt.minimize(obj)

        # CBF for static obstacles
        for i in range(self.N):
            for j in range(self.n_SO):            
                st = self.opt_states[i,:]
                st_next = self.opt_states[i+1,:]
                h = (st[0]-self.SO[j,0])**2+(st[1]-self.SO[j,1])**2-(self.R_husky+self.SO[j,2])**2
                h_next = (st_next[0]-self.SO[j,0])**2+(st_next[1]-self.SO[j,1])**2-(self.R_husky+self.SO[j,2])**2
                self.opt.subject_to(h_next-(1-self.cbf_gamma)*h >= 0) 

        # constraint the change of velocity
        for i in range(self.N-1):
            dvel = (self.opt_controls[i+1,:] - self.opt_controls[i,:])/self.DT
            self.opt.subject_to(self.opt.bounded(-self.max_dv, dvel[0], self.max_dv))
            self.opt.subject_to(self.opt.bounded(-self.max_domega, dvel[1], self.max_domega))

        # boundary of state and control input
        self.opt.subject_to(self.opt.bounded(self.min_x, x, self.max_x))
        self.opt.subject_to(self.opt.bounded(self.min_y, y, self.max_y))
        self.opt.subject_to(self.opt.bounded(self.min_theta, theta, self.max_theta))    
        self.opt.subject_to(self.opt.bounded(self.min_v, v, self.max_v))
        self.opt.subject_to(self.opt.bounded(self.min_omega, omega, self.max_omega))
        
        # setup optimization parameters
        opts_setting = {'ipopt.max_iter':200,'ipopt.print_level':0,'print_time':0,'ipopt.acceptable_tol':1e-8,'ipopt.acceptable_obj_change_tol':1e-6}
        #max iter was 2000
        self.opt.solver('ipopt', opts_setting)
    
    def solve(self, next_trajectories, next_controls):
        
        self.opt.set_value(self.x_ref, next_trajectories)       # update feedback state and reference
        self.opt.set_value(self.u_ref, next_controls)           # update feedback control and reference
        
        self.opt.set_initial(self.opt_states, self.next_states) # provide the initial guess of state for the next step
        self.opt.set_initial(self.opt_controls, self.u0)        # provide the initial guess of control for the next step       
        ## solve the problem
        sol = self.opt.solve()
        
        if sol.stats()['return_status'] != 'Solve_Succeeded':
            kenny_loggins(f"[NPMC-solver]: ERROR! Solver return status: {sol.stats()['return_status']}")      
        
        ## obtain the control input
        new_u0 = sol.value(self.opt_controls)
        self.u0[:-1, :] = new_u0[1:, :]
        self.u0[-1, :] = new_u0[-1, :]
        self.next_states = sol.value(self.opt_states)
        return new_u0[0,:]

    def reset_nmpc(self, obstacle, cbf_gamma):               # Reset the NMPC for the next episode
        self.u0 = np.zeros((N, 2))                                          # Reset NMPC internal control variable !! Must do this when resetting the episode or the NMPC will cry
        self.next_states = np.zeros((N+1, 3))                               # Reset NMPC internal state variable  !!  Must do this when resetting the episode or the NMPC will cry
        self.SO = obstacle                                                  # Set the obstacle parameter for NMPC
        self.cbf_gamma = cbf_gamma                                          # Set the CBF parameter for NMPC
        self.n_SO = len(obstacle[:, 0])                                     # Set the number of obstacles (should always be 1 for this example)
        self.setup_controller()                                             # Setup the controller optimisation for the next episode
        return

class RosbagRecorder:
    def __init__(self, cbf_gamma, obstacle, target):
        self.rosbag_process = None                  # Initialize rosbag process
        self.rosbag_run = 0                         # Initialize rosbag run number
        self.rosbag_name = None                     # Initialize rosbag name
        self.rec_dir = '/home/user/husky/rosbag/'   # Base Directory to save rosbag files
        self.dir_id = self.init_dir()               # Initialize the directory for rosbag files
        self.cbfgamma = cbf_gamma                   # Initialize CBF parameter
        self.obstacle = obstacle.flatten().tolist() # Initialize obstacle parameter
        self.target = target.tolist()               # Initialize target parameter

    def init_dir(self):
        if not os.path.exists(self.rec_dir):                    # Check if the base directory exists
            kenny_loggins("[NMPC-rosbag]: ERROR! Rosbag Directory does not exist")       # debug
            exit()                                               # Exit if the directory does not exist
        directories = [d for d in os.listdir(self.rec_dir) if os.path.isdir(os.path.join(self.rec_dir, d))]     # Get the list of directories
        if directories:                                                                                         # If directories exist
            most_recent_dir = max(directories, key=lambda d: os.path.getmtime(os.path.join(self.rec_dir, d)))       # Get the most recent directory
            next_dir = int(most_recent_dir) + 1                                                                     # Get the next directory id
        else:                                                                                                   # If no directories exist
            next_dir = 1                                                                                            # Set the next directory id to 1                                                
        self.rec_dir = os.path.join(self.rec_dir, str(next_dir).zfill(3))                                       # Create the next directory                
        os.makedirs(self.rec_dir)                                                                               # Make the next directory
        csv_file_path = os.path.join(self.rec_dir, 'run_info.csv')
        with open(csv_file_path, 'w') as csv_file:
            pass  # Create an empty .csv file


        return next_dir

    def new_run(self):
        self.rosbag_run += 1
        self.rosbag_name = self.rec_dir + '/nmpc_run_' + str(self.rosbag_run).zfill(6)
        if self.rosbag_run == 1:
            header = ['RunID', 'cbf_gamma', 'obs_x', 'obs_y', 'obs_rad', 'tgt_x', 'tgt_y', 'tgt_w', 'reward', 'ep_state']
            with open(os.path.join(self.rec_dir, 'run_info.csv'), 'a') as csv_file:
                csv_file.write(','.join(map(str, header)) + '\n')

    def write_info(self,reward,ep_state):
        this_run_info = [self.rosbag_run, self.cbfgamma] + self.obstacle + self.target + [reward, ep_state]
        with open(os.path.join(self.rec_dir, 'run_info.csv'), 'a') as csv_file:
            csv_file.write(','.join(map(str, this_run_info)) + '\n')



    def start_recording(self):
        self.new_run()                                                              # Make new directory for run
        if self.rosbag_process is None:                                             # Start rosbag recording
            self.rosbag_process = subprocess.Popen(['rosbag', 'record', '-q' , '-O', self.rosbag_name, '/odometry/filtered', '/cmd_vel'])
            rospy.sleep(0.5)

    def stop_recording(self, ep_state):
        
        # if cbf_gamma is None or obstacle is None or target is None:                 # If episode info is not provided   
        #     ep_info = [-1.0, -1.0, -1.0]                                              # Create episode info vector
        # else:
        #     ep_info = [cbf_gamma] + obstacle.flatten().tolist() + target.tolist()       # Create episode info vector
        # pub_info.publish(Float32MultiArray(data=ep_info))                           # Publish episode info for rosbag
        # rospy.sleep(1)                                                            # Wait for the message to be published
        if self.rosbag_process is not None:
            try:
                self.rosbag_process.send_signal(signal.SIGINT)                      # Send SIGINT to rosbag to stop recording
                rospy.sleep(0.5)                                                    # Wait for rosbag to stop
                self.rosbag_process.wait(timeout=4)                                 # Wait for process to terminate
            except subprocess.TimeoutExpired:                                     # If timeout occurs
                kenny_loggins("[NMPC-rosbag]: rosbag process didn't terminate in time, forcing shutdown.")
                self.rosbag_process.terminate()                                     # Terminate the process if it doesn't stop
            finally:
                self.rosbag_process = None                                              # Set rosbag process to None to reset
            
            kenny_loggins("[NMPC-rosbag]: Stopped rosbag recording")                    # Log info to console
            
            if ep_state != -1.0:                                                        # If stop is not due to exit
                reward = self.assess_episode(self.cbfgamma, self.obstacle, self.target)     # Assess the episode to return reward
            else:
                reward = -1.0                                                               # Set reward to -1 if episode is stopped due to exit
            print(f"\n\n\nCBF: {self.cbfgamma} \nObstacle: {self.obstacle[2]}m \nReward: {reward}\n\n\n")
            self.write_info(reward,ep_state)                                            # Write the episode info to the csv file
            return reward
        
    def assess_episode(self, cbf_gamma, obstacle, target):       # Assess the episode for the next scenario
        rospy.sleep(1)
        bagfile = self.rosbag_name + '.bag'             # Get the bag file name
        bag = rosbag.Bag(bagfile)                       # Open the bag file
        # print(cbf_gamma,obstacle,target)
        reward = get_reward(bag,cbf_gamma,obstacle,target)                   # Set the reward for the episode
        return reward

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

# ###################################################################################################
# FUNCTIONS                                                                                           
# ###################################################################################################

def exit_handler():         # Exit handler for the node
    kenny_loggins("[NMPC-Exiting]: Shutting down Husky NMPC Node...")             # Log message to console
    ep_record.stop_recording(ep_state=-1.0)                                          # Stop recording the episode

def kenny_loggins(msg, logto=0, lvl=None):  # Danger Zone
    if logto == 0:                          # Print log messages to console
        print(msg)

    elif logto == 1:                        # Log to ros log # add log levels
        rospy.logdebug(msg)

def wraptopi(x):                            # used to wrap angle errors to interval [-pi pi]
    pi = np.pi  
    x = x - np.floor(x/(2*pi)) *2*pi
    if x > pi:
        x = x - 2*pi
    return x
  
odom = Odometry()
ofleg = False
def odom_callback(msg):                         # Feedback state callback
    global odom, ofleg
    odom = msg
    ofleg = True

def getyaw(odom):                               # Convert quaternion to euler
    orientation = odom.pose.pose.orientation
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    roll, pitch, yaw = euler_from_quaternion(q)
    wrap_yaw = wraptopi(yaw)
    return wrap_yaw, yaw

def desired_trajectory(curent_state, N, target_state):              # Generate st_ref and ct_ref at each sampling time from desired trajectory vector
    state_vector = np.vstack((curent_state, 
                              np.tile(target_state, (N+1, 1))))     # Generate the state vector
    control_vector = np.zeros((N, 2))                               # Generate the control vector              
    return state_vector, control_vector

def node_startup():         # Start ROS node and wait for odometry feedback
    global ofleg, pub_vel, pub_response, pub_info, husky_radius, request
    husky_radius = 0.55                                                     # Husky robot clearance radius (tight)
    rospy.init_node("robot_nmpc_cbf_node", anonymous=True)                  # Init ROS node
    rospy.Subscriber('/odometry/filtered', Odometry, odom_callback)         # Subscribe to feedback state
    request = topicQueue('/request', Float32MultiArray)                     # Create a queue for request topic
    pub_vel=    rospy.Publisher('/cmd_vel', Twist, queue_size=5)            # Publish control input from nmpc
    # pub_hb=     rospy.Publisher('/zheartbeat', Int16, queue_size=5)          # Publish heartbeat signal (mpc processing time)
    # pub_tsep=   rospy.Publisher('/ztarget_sep', Float32, queue_size=5)       # Publish distance to target
    # pub_ssep=   rospy.Publisher('/zobstacle_sep', Float32, queue_size=5)     # Publish distance to obstacle
    # pub_status= rospy.Publisher('/zep_status', Int8, queue_size=5)              # Publish status of the node (0: running, 1: reached target, 2: collision)
    pub_response=rospy.Publisher('/response', Float32MultiArray, queue_size=10)
    pub_info=   rospy.Publisher('/ep_info', Float32MultiArray, queue_size=5)

    rate = 10  
    r = rospy.Rate(rate)                                    # Set Rate of the node in Hz
    
    kenny_loggins("[NMPC]: Starting Husky NMPC Node...")             # Wait for odometry feedback
    while(ofleg == False):
        time.sleep(1)
        kenny_loggins("[NMPC]: Waiting for odometry feedback...")
    kenny_loggins("[NMPC]: Husky NMPC Node is ready!!!") 
    time.sleep(1)
    kenny_loggins("[NMPC]: Start NMPC simulation!!!")
    return r

def state_feedback(odom):    # Read feedback state from odometry
    x_fb = odom.pose.pose.position.x    # Get feedback position
    y_fb = odom.pose.pose.position.y
    theta_fb, _ = getyaw(odom)          # Get the wrapped yaw angle
    pos_fb = np.array([x_fb, y_fb, theta_fb])
    return pos_fb

def assess_if_done(target, pos_fb, obstacle,ep_state):   # Assess if the episode is done (target reached or collision)}:
    # episode status states (0: waiting to start, 1: running, 2: at target, 3: collision)
    
    collision_tol = 0.01                                    # Collision tolerance
    target_tol = 0.2                                        # Target tolerance
    tgt_sep = np.linalg.norm(pos_fb[:2] - target[:2])       # Distance to the target
    start_sep = np.linalg.norm(pos_fb[:2])                  # Distance to the start point
    obs_sep = np.linalg.norm(pos_fb[:2] - obstacle[0,:2])   # Obstacle-vehicle center separation
    obs_sep = obs_sep - obstacle[0,2] - husky_radius                # Safety distance between vehicle and obstacle (center sep - obstacle radius - husky radius)
    # pub_tsep.publish(tgt_sep)                               # Publish the distance to the target
    # pub_ssep.publish(obs_sep)                               # Publish the distance to the obstacle
    new_state = ep_state                                    # Initialize the new episode state
    # if ep_state==0 and start_sep < 0.5:             # If the vehicle is close to the start point
    #     new_state = 1                   # Set the episode state to running
    
    if ep_state == 1:                   # When running
        if obs_sep <= collision_tol:        # If collision with obstacle
            new_state = 3                       # Set the episode state to collision
        elif tgt_sep < target_tol:          # Otherwise if within tolerance of target position
            new_state = 2                       # Set the episode state to target reached
    
    # pub_status.publish(new_state)   # Publish the episode state
    is_done = new_state > 1          # If the episode state is not start (0) or running (1), then it is done

    if is_done:                     # If the episode is done
        kenny_loggins("[NMPC-Done]: Episode Done!! | last state: " + str(ep_state) + " new state: " + str(new_state)  )                                         # debug

    return new_state, is_done

def trainer_request():      # Request the next episode from the trainer
    kenny_loggins("\n\n\n >>>>> [NMPC-NextEp]: Getting next episode")                    # debug
    pub_response.publish(Float32MultiArray(data=[-1.0, 0.0, 0.0]))          # send response with -1 to get next episode from trainer
    # kenny_loggins("[NMPC-NextEp]: Waiting for next episode")                # debug
    # next_episode = rospy.wait_for_message('/request', Float32MultiArray)    # wait for response from trainer
    # next_episode = np.array(next_episode.data)                              # convert to numpy array
    waitfleg = False
    while request.is_empty():                                               # wait for response from trainer               
        rospy.sleep(1)                                                        # sleep for 0.1s
        if not waitfleg:
            kenny_loggins("[NMPC-NextEp]: Waiting for next episode")                # debug
            waitfleg = True
    next_episode = request.pop()                                            # pop the response from the queue                       
    if next_episode[0] > 0:                                                 # check if next episode is valid                             
        rounded_episode = tuple(round(elem, 4) for elem in next_episode)
        kenny_loggins("[NMPC-NextEp]: Next episode received: " + str(rounded_episode))               # debug
        cbf_parm = round(float(next_episode[0]),4)
        obs_rad = round(float(next_episode[1]),4)
        # rospy.set_param('/cbf_gamma', cbf_parm)           # Set CBF parameter
        # rospy.set_param('/obstacle', obs_rad)            # Set obstacle radius
    else:
        kenny_loggins("[NMPC-NextEp]: Error getting next episode")                      # debug

    return cbf_parm, obs_rad

def setup_scenario():                               # Setup the scenario for episode
    cbf_gamma, orad = trainer_request()                 # Get the CBF parameter and obstacle radius from the trainer for next episode
    approach_sep = 10                                   # Approach separation from the obstacle
    target_sep = 5                                      # Target separation from the obstacle
    obstacle_x = husky_radius + orad + approach_sep     # Calculate the obstacle x position
    target_x =   obstacle_x + orad + target_sep         # Calculate the target x position
    obstacle = np.array([[obstacle_x, -0.05, orad]])    # define obstacle [ x , y , radius ]
    target = np.array([target_x, 0, 0])                 # define target [ x , y , theta ]
    # rospy.set_param('/obs_info', obstacle.tolist())     # Set the obstacle parameter
    # rospy.set_param('/tgt_info', target.tolist())       # Set the target parameter
    kenny_loggins("\n[NMPC-SetEp]: New scenario | cbf_gamma: " + str(cbf_gamma) + ' obs: ' + str(obstacle) + ' tgt: ' +str(target) )
    return cbf_gamma, obstacle, target

def reset_simulation():     # Reset the simulation for the next episode
    call_reset = subprocess.run(['rosrun', 'cbf_rl_train', 'reset_episode.py'])
    kenny_loggins("[NMPC-Reset]: Reset Complete")           # debug

# ###################################################################################################
# MAIN FUNCTION                                                                                           
# ###################################################################################################

def nmpc_node():                    # Main function to run NMPC
    global nmpc, N, run_id, ep_record
    atexit.register(exit_handler)   # Register exit handler for the node
    r = node_startup()              # Start node 
    pos_fb = state_feedback(odom)   # Read the initial feedback state    
    DT = 0.1                        # Set the timestep for NMPC
    N = 30                          # Set the horizon length
    # NMPC constraints and weights
    min_x = -100            # workspace limitations
    max_x = 100
    min_y = -100 
    max_y = 100
    min_theta = -np.inf     # angle limitations
    max_theta = np.inf
    min_v = -1.0            # velocity limitations
    max_v = 1.0
    min_omega = -0.7854     # angular velocity limitations
    max_omega = 0.7854

    limitation = [min_x, max_x, min_y, max_y, min_theta, max_theta, min_v, max_v, min_omega, max_omega]
    
    W_q = np.diag([5.0, 5.0, 0.5])                  # weights for states
    W_r = np.diag([5.0, 0.1])                       # weights for controls
    # W_v = 10**5*np.diag([1.0, 1.0, 0.000001])       # weights for terminal state
    W_v = 10**4*np.diag([1.0, 1.0, 0.001])       # weights for terminal state
    
    # while not rospy.get_param('/zenable', False):
    #     rospy.sleep(1)
    rospy.sleep(2)
    cbf_gamma, obstacle, target = setup_scenario()   # Setup the scenario for the initial episode
    nmpc = NMPC_CBF_Terminal(pos_fb, limitation, DT, N, W_q, W_r, W_v, cbf_gamma, obstacle)   # Create NMPC object
    run_id = 1
    ep_state = 1                                    # Episode state (0: waiting to start, 1: running, 2: at target, 3: collision)
    vel_msg = Twist()                               # Initialize the velocity message
    done = False                                    # Episode done flag
    ep_record = RosbagRecorder(cbf_gamma,obstacle,target)                    # Create rosbag recorder object
    ep_record.start_recording()                     # Start recording the episode

    while not rospy.is_shutdown():
        
        if ep_state == 0:                                       # If the episode is waiting to start from previous reset
            cbf_gamma, obstacle, target = setup_scenario()          # Setup the scenario for the next episode
            nmpc.reset_nmpc(obstacle, cbf_gamma)                    # Reset the NMPC for the next episode
            ep_record.cbfgamma = cbf_gamma                         # Set the CBF parameter for recording
            ep_record.obstacle = obstacle.flatten().tolist()        # Set the obstacle parameter for recording      
            ep_record.target = target.tolist()                      # Set the target parameter for recording
            ep_record.start_recording()                             # Start recording the episode
            ep_state = 1                                            # Set the episode state to running
        
        pos_fb = state_feedback(odom)                                           # Read feedback state
        ep_state, done = assess_if_done(target, pos_fb, obstacle, ep_state)     # Assess state of episode (target reached or collision)
        
        if not done:                                                            # Run the NMPC until the episode is done
            next_traj, next_cons = desired_trajectory(pos_fb, N, target)            # Generate the desired trajectory and control input
            # tic = time.time()                                                     # Start the timer
            try:                                                                    # Try to solve the NMPC problem                                 
                vel = nmpc.solve(next_traj, next_cons)                                  # Solve the NMPC problem
                # toc = (time.time() - tic)*1000                                          # Calculate the processing time in ms
                # toc = min(int(toc), 5000)                                               # Limit the processing time to 5000ms
                # pub_hb.publish(toc)                                                     # Publish the processing time to heartbeat topic
            except:
                kenny_loggins("[NMPC-Solve]: ERROR! NMPC Solver failed")
                print(next_traj)
                print(next_cons)
                print(ep_state)
                vel = [0.0, 0.0]                                                        # Set the control input to zero if NMPC fails
            vel_msg.linear.x = vel[0]                                               # Set the linear veolcity                 
            vel_msg.angular.z = vel[1]                                              # Set the angular velocity
            pub_vel.publish(vel_msg)                                                # Publish the control input to husky

        else:                                                                # If done, reset the simulation for next episode
            # pub_hb.publish(-1)                                                       # Publish zero processing time for done episode
            reward = ep_record.stop_recording(ep_state)          # Stop recording the episode
            reset_simulation()                                                      # Reset the simulation for the next episode
            ep_state = 0                                                            # Set the episode state to waiting to start
        
        r.sleep()                                                               # Sleep for the rest of the time

if __name__ == '__main__':    
    try:
        nmpc_node()
    except rospy.ROSInterruptException:
        pass