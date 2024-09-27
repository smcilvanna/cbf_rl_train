import numpy as np
import rosbag
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32MultiArray
import imageio
import os

def wraptopi(x):                            # used to wrap angle errors to interval [-pi pi]
    pi = np.pi  
    x = x - np.floor(x/(2*pi)) *2*pi
    if x > pi:
        x = x - 2*pi
    return x
  
def getyaw(odom):                               # Convert quaternion to euler
    orientation = odom.pose.pose.orientation
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    roll, pitch, yaw = euler_from_quaternion(q)
    wrap_yaw = wraptopi(yaw)
    return wrap_yaw, yaw

def check_ep_info(dirid=None):
    if dirid == None:
        print("Directory ID not provided")
        return

    dir_to_plot = '/home/user/husky/rosbag/' + str(dirid).zfill(3)                  # directory containing the bag files

    for file_name in sorted(os.listdir(dir_to_plot)):                           # list the files in the directory in order      
        continue                                                                           

    lastid = int(file_name.split('_')[2].split('.')[0])                              # get the last file id
    input(f"Last File ID: {lastid}  | Press Enter to continue...")              # wait for user to confirm

    for idx in range(1, lastid+1):
        bagfile = dir_to_plot + '/nmpc_run_' + str(idx).zfill(6) + '.bag'       # bag file to read   
        bag = rosbag.Bag(bagfile)

        episode_info = None
        for topic, msg, t in bag.read_messages(topics=['/ep_info']):
            episode_info = msg.data                                     # [cbf_gamma, obs_x, obs_y, obs_r, target_x, target_y, target_w]
        
        if episode_info == None:
            print(" >>>>>>>>>>>>>>> Episode info not found for file index ", idx)
        else:
            print([round(info, 4) for info in episode_info])
 
def add_episode_info(dirid=None, index=None, cbf_gamma=None, orad=None):
    if dirid == None or index == None:
        print("Directory ID or File Index not provided")
        return

    if cbf_gamma == None or orad == None:
        print("CBF Gamma or Obstacle Radius not provided")
        return

    test_data = [cbf_gamma, orad]
    obs_x, tgt_x = setup_scenario(test_data)
    ep_info = [cbf_gamma, obs_x, -0.05, orad, tgt_x, 0, 0]
    print("\n")
    print(ep_info)
    bagfile = f'/home/user/husky/rosbag/{str(dirid).zfill(3)}/nmpc_run_{str(index).zfill(6)}.bag'
    input(f"Bag file: {bagfile}  | Press Enter to continue...")              # wait for user to confirm
    bag = rosbag.Bag(bagfile, 'a')
    ep_info_msg = Float32MultiArray(data=ep_info)
    bag.write('/ep_info', ep_info_msg)
    bag.close()

def get_episode_info(bag):                                          #old function, use csv file instead
    for topic, msg, t in bag.read_messages(topics=['/ep_info']):
        episode_info = msg.data                                     # [cbf_gamma, obs_x, obs_y, obs_r, target_x, target_y, target_w]
    
    print(episode_info)
    cbf_gamma = round(episode_info[0],4)
    obstacle = [round(elm,4) for elm in episode_info[1:4]]
    target = [round(elm,4) for elm in episode_info[4:]]

    return cbf_gamma, obstacle, target

def get_episode_info_csv(dirid):
    csv_file = f'/home/user/husky/rosbag/{str(dirid).zfill(3)}/run_info.csv'
    with open(csv_file, 'r') as f:    
        header = f.readline().strip().split(',')    # Read the header row
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1, dtype=float)
    return data, header

def get_odom(bag):
    X = bag.read_messages(topics=['/odometry/filtered'])
    odom = np.empty((0,3))
    for topic, msg, t in X:
        yaw = getyaw(msg)[0]
        # x = np.array( [[msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]] )
        x = np.array( [[msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]] )
        odom = np.append(odom, x, axis=0)
        # odom = np.round(odom, 3)
    return odom

def calculate_distance(odom, obstacle, target):     # calculate the distance travelled from the odom data
    
    positions= odom[:,0:2]                          # x, y          
    diffs= np.diff(positions, axis=0)               # calculate difference between each consecutive position
    distances= np.sqrt(np.sum(diffs**2, axis=1))    # calculate distance between each consecutive position
    total_distance= np.sum(distances)
    
    # estimate the optimal distance
    vrad = 0.55
    orad = obstacle[2]
    ox = obstacle[0]
    gx = target[0]
    p1 = np.sqrt( (ox)**2       + (orad + vrad)**2 )            # first leg distance  
    p2 = np.sqrt( (gx - ox)**2  + (orad + vrad)**2 ) - 0.2      # second leg distance - end tolerances
    p3 = 0
    est_opt_dist = p1 + p2 + p3

    # calculate the distance from the obstacle and target
    end_pos = positions[-1]
    fin_sep = np.sqrt( (end_pos[0] - target[0])**2 + (end_pos[1] - target[1])**2 )
    fin_yaw = odom[-1,2]

    # check for collision
    safe_tol = 0.05
    centre_sep = np.sqrt((positions[:, 0] - obstacle[0])**2 + (positions[:, 1] - obstacle[1])**2)
    safe_sep = centre_sep - obstacle[2] - vrad
    collision = np.any(safe_sep < safe_tol)

    return total_distance, est_opt_dist, fin_sep, fin_yaw, collision

def get_reward(bag, cbf_gamma, obstacle, target):

    odom = get_odom(bag)
    dist, opt_dist, fin_sep, fin_yaw, collision = calculate_distance(odom, obstacle, target)
    reward = 0.0
    if collision:
        reward = -100
    elif fin_sep < 0.5:
        reward = opt_dist/dist * 100
    return reward

def check_bag(bag):                             # check if all expected topics are present in the bag file
    expected_topics = ['/cmd_vel',                  # topics that should be present in the bag file
                       '/odometry/filtered', 
                       '/zep_status', 
                       '/zheartbeat', 
                       '/zobstacle_sep', 
                       '/ztarget_sep']
    all_topics = True                               # flag to check if all expected topics are present in the bag file
    topics_info = bag.get_type_and_topic_info()[1]  # get the topics present in the bag file

    for expected_topic in expected_topics:
        # print("\n\n")
        # print(expected_topic)
        topic_exists = expected_topic in topics_info
        # print(topic_exists)
        all_topics = all_topics and topic_exists

    return all_topics

def plot_run(bag, test=[-1,-1], reward=0.00, reward2=None):                                     # plot the run from the bag file
    husky_radius = 0.55                                                         # Husky robot radius
    gamma = test[0]                                                             # Get the CBF parameter 
    orad = test[1]                                                              # Get the obstacle radius
    crad = orad + husky_radius                                                  # Calculate the clearance radius
    obs_x, target_x = setup_scenario(test)                                      # Get obstacle and target positions for the test scenario
    print(f"Obstacle: {obs_x} Target: {target_x}")                              # Print the obstacle and target positions
    odom = get_odom(bag)                                                        # Get the odometry data from the bag file
    fig, ax = plt.subplots()                                                    # Create a figure and axis                          
    ax.plot(odom[:, 0], odom[:, 1], label='Trajectory')                         # plot the trajectory
    # ax.scatter(odom[0, 0], odom[0, 1], color='green', label='Start')          # plot the start point
    ax.scatter(target_x, 0, color='green', label='Target', marker='x', s=90)    # plot the target point
    ax.scatter(odom[-1, 0], odom[-1, 1], color='red', label='End')              # plot the end point
    obstacle = plt.Circle((obs_x, 0), orad, color='grey', fill=False, linestyle='--', label='Obstacle')                     # plot the obstacle
    ax.add_artist(obstacle)                                                                                                 # add the obstacle to the plot
    obstacle_clearance = plt.Circle((obs_x, 0), crad, color='red', fill=False, linestyle='-', label='Obstacle Clearance')  # plot the obstacle clearance
    ax.add_artist(obstacle_clearance)                                                                                       # add the obstacle clearance to the plot
    
    # Determine the color based on the reward value
    if reward < 0:
        color = 'red'
    elif reward > 0:
        color = 'green'
    else:
        color = 'black'

    rwtxt = ax.text(0.6, 0.125, f'Reward: {reward:.2f}', fontsize=12, color=color, transform=ax.transAxes)  # add the reward text to the plot
    ax.add_artist(rwtxt)                                                                    # add the reward text to the plot
    if reward2 != None:
        rwtxt2 = ax.text(0.6, 0.075, f'Reward: {reward2:.2f}', fontsize=12, color='blue', transform=ax.transAxes)
        ax.add_artist(rwtxt)
    
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f'Trajectory Plot  | Gamma: {gamma:.2f}  | Obs-Rad: {orad}')
    # ax.axis('equal')
    ax.set_xlim([0, 40])
    ax.set_ylim([-20, 20])
    ax.legend()
    ax.grid(True)
    return fig

def bag_topics(bag):    
    topic_dic = bag.get_type_and_topic_info()[1]
    keys = topic_dic.keys()
    msgs = len(topic_dic.keys())
    return keys, msgs

def read_bag(bagfile):
    bag = rosbag.Bag(bagfile)
    print(bag)

    cbf_gamma, obstacle, target = get_episode_info(bag)
    odom = get_odom(bag)

    dist, opt_dist = calculate_distance(odom, obstacle, target)
    
    print('cbf_gamma:', cbf_gamma)
    print('obstacle:', obstacle)
    print('target:', target)
    print('Distance travelled:', dist)
    print('Optimal Distance: ' , opt_dist)

    reward = opt_dist/dist * 100
    print('Reward:', reward)


    # U = bag.read_messages(topics=['/cmd_vel'])
    # X = bag.read_messages(topics=['/odometry/filtered'])
    # ctrls = np.empty((0,2))
    # for topic, msg, t in U:
    #     u = np.array( [[msg.linear.x, msg.angular.z]] )
    #     ctrls = np.append(ctrls, u, axis=0)

    # print(np.shape(ctrls))
    # print(ctrls)

def get_test_list():
    # issue with rosbag not recording the test scenario
    # manually define the test scenarios (copy from dummy trainer)
    # cbf_gammas = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    # obs_radii = np.array([1.0, 5.0])

    # cbf_gammas = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75])
    # obs_radii = np.array([5.0])

    cbf_gammas = np.array([0.5, 0.75])
    obs_radii = np.array([5.0])

    grid1, grid2 = np.meshgrid(cbf_gammas, obs_radii)
    combinations = np.column_stack([grid1.ravel(), grid2.ravel()])
    return combinations

def setup_scenario(test):                          # Get obstacle and target positions for the test scenario 
    husky_radius = 0.55                                 # Husky robot radius
    orad = test[1]                                      # Get the CBF parameter and obstacle radius from the trainer for next episode
    approach_sep = 10                                   # Approach separation from the obstacle
    target_sep = 5                                      # Target separation from the obstacle
    obstacle_x = husky_radius + orad + approach_sep     # Calculate the obstacle x position
    target_x =   obstacle_x + orad + target_sep         # Calculate the target x position
    return obstacle_x, target_x

def create_gif_from_images(framerate=1.0, image_dir='/home/user/husky/husky_mpc_ws/src/cbf_rl_train/src/.temp/rosbag/temp_plts', 
                           output_gif='/home/user/husky/husky_mpc_ws/src/cbf_rl_train/src/.temp/rosbag/trajectory.gif'):
    images = []
    for file_name in sorted(os.listdir(image_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(image_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(output_gif, images, fps=framerate)

def plot_dir(dirid=1,framerate=1.0):
    dir_to_plot = '/home/user/husky/rosbag/' + str(dirid).zfill(3)              # directory containing the bag files

    print(f"Contents of {dir_to_plot}:")                                        # print the contents of the directory
    for file_name in sorted(os.listdir(dir_to_plot)):                           # list the files in the directory in order      
        print(file_name)                                                            # print the file name               

    lastid = int(file_name.split('_')[2].split('.')[0])                         # get the last file id
    input(f"Last File ID: {lastid}  | Press Enter to continue...")              # wait for user to confirm

    run_data, run_header = get_episode_info_csv(dirid)                         # get the episode information from the csv file

    for idx in range(1, lastid+1):
        bagfile = dir_to_plot + '/nmpc_run_' + str(idx).zfill(6) + '.bag'       # bag file to read
        bag = rosbag.Bag(bagfile)                                               # open the bag file
        print(f"\n\nBag file: {bagfile}")                                       # print the bag file name          
        cbf_gamma, obstacle, target = get_episode_info(bag)                     # get the episode information
        test_info = [cbf_gamma, obstacle[2]]                                    # test information
        reward = get_reward(bag, cbf_gamma, obstacle, target)                   # get the reward
        reward2 = float(run_data[idx-1, 8])                                     # get the reward from the csv file
        fig = plot_run(bag, test_info,reward, reward2)                          # plot the run
        output_file = f'{out_dir}/nmpc_run_{str(idx).zfill(6)}.png'             # output file name
        fig.savefig(output_file)                                                # save the plot to a file in temp folder
        plt.close(fig)                                                          # close the plot              

    create_gif_from_images(framerate=framerate)                                                     # create a gif from the images           

def set_and_check_output_dirs(out_dir=None):
    dirOK = True
    if out_dir == None:                                                             # check if the output directory is provided
        out_dir = ('/home/user/husky/husky_mpc_ws/src/'
                    'cbf_rl_train/src/.temp/rosbag/temp_plts')                     # set the default output directory
    if not os.path.exists(out_dir):                                                 # check if the output directory exists
        print("Output directory does not exist.")                                       # print message if the output directory does not exist
        dirOK = False                                                                   # set the flag to False                            

    png_files = [f for f in os.listdir(out_dir) if f.endswith('.png')]              # get the list of .png files in the output directory
    if png_files:                                                               # check if there are no .png files in the output directory                                                                                            # if there are .png files in the output
        print(f"Found {len(png_files)} .png files in the output directory.")            # print the number of .png files found in the output directory
        dirOK = False                                                                   # set the flag to False                                

    return dirOK, out_dir

if __name__ == '__main__':

    dirok, out_dir = set_and_check_output_dirs()
    if not dirok:
        print("Output directory is not empty. Exiting...")
        exit()

    # plot_dir(dirid=10, framerate=2.0)
    # check_ep_info(10)
    # add_episode_info(dirid=10, index=30, cbf_gamma=0.75, orad=3.0)

    
