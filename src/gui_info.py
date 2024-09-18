#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
import tkinter as tk

class GuiInfo:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('gui_info', anonymous=True)
        
        # Create the Tkinter window
        self.root = tk.Tk()
        self.root.title("GUI Info")
        self.root.attributes('-topmost', True)      # Keep the window on top
        
        # Labels to display the values
        self.ztarget_label = tk.Label(self.root, text="Target Separation: N/A")
        self.ztarget_label.pack()

        self.zobstacle_label = tk.Label(self.root, text="Obstacle Separation: N/A")
        self.zobstacle_label.pack()

        # ROS subscribers
        rospy.Subscriber("/ztarget_sep", Float32, self.target_callback)
        rospy.Subscriber("/zobstacle_sep", Float32, self.obstacle_callback)

        # Call the update function periodically
        self.update_gui()

    def target_callback(self, data):
        # Update target separation label
        self.ztarget_label.config(text="Target Separation: {:.2f}".format(data.data))

    def obstacle_callback(self, data):
        # Update obstacle separation label
        self.zobstacle_label.config(text="Obstacle Separation: {:.2f}".format(data.data))

    def update_gui(self):
        # Update the GUI every 100 ms
        self.root.update()
        if not rospy.is_shutdown():
            self.root.after(100, self.update_gui)

if __name__ == '__main__':
    try:
        gui_info = GuiInfo()
        gui_info.root.mainloop()
    except rospy.ROSInterruptException:
        pass
