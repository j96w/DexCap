"""
Visualize saved pcd file and use keyboard to place it on the robot table

Example usage:
(1) for all frames
python transform_to_robot_table.py --directory ./saved_data/
"""

import argparse
import os
import copy
import zmq
import cv2
import sys
import shutil
import open3d as o3d
import numpy as np
import platform
from pynput import keyboard
from transforms3d.quaternions import qmult, quat2mat
from transforms3d.axangles import axangle2mat
from scipy.spatial.transform import Rotation
from transforms3d.euler import quat2euler, mat2euler, quat2mat, euler2mat
from visualizer import *

def on_press(key):
    global next_frame, delta_movement_accu, delta_ori_accu, delta_movement_accu_left, delta_ori_accu_left, adjust_movement, adjust_right, frame, step, fixed_transform, dir_path

    # Determine the OS
    os_type = platform.system()
    assert os_type == "Windows"

    # Windows-specific key bindings are handled in the AttributeError section
    if key == keyboard.Key.right:  # x negative
        fixed_transform[0] += step
    elif key == keyboard.Key.left:  # x positive
        fixed_transform[0] += -step
    elif key == keyboard.Key.up:  # y positive
        fixed_transform[1] += step
    elif key == keyboard.Key.down:  # y negative
        fixed_transform[1] += -step
    elif key == keyboard.Key.home:  # z positive
        fixed_transform[2] += step
    elif key == keyboard.Key.page_up:  # z negative
        fixed_transform[2] += -step
    elif key == keyboard.Key.space:
        np.savetxt(os.path.join(dir_path, "map_to_robot_table_trans.txt"), fixed_transform)

class TableDataVisualizer(DataVisualizer):
    def __init__(self, directory, table_frame):
        super().__init__(directory)

        self.table_frame = table_frame

    def replay_all_frames(self):
        """
        Visualize all frames continuously
        """
        try:
            with keyboard.Listener(on_press=on_press) as listener:
                if self.R_delta_init is None:
                    self.initialize_canonical_frame()

                frame = 0
                first_frame = True
                while True:
                    if not self._load_frame_data(frame, load_table_points=True):
                        break

                    if first_frame:
                        self.vis.add_geometry(self.table_pcd)
                        self.vis.add_geometry(self.pcd)
                        self.vis.add_geometry(self.coord_frame_1)
                        self.vis.add_geometry(self.coord_frame_2)
                        self.vis.add_geometry(self.coord_frame_3)
                        for joint in self.left_joints + self.right_joints:
                            self.vis.add_geometry(joint)
                        for cylinder in self.left_line_set + self.right_line_set:
                            self.vis.add_geometry(cylinder)
                    else:
                        self.vis.update_geometry(self.pcd)
                        self.vis.update_geometry(self.coord_frame_1)
                        self.vis.update_geometry(self.coord_frame_2)
                        self.vis.update_geometry(self.coord_frame_3)
                        for joint in self.left_joints + self.right_joints:
                            self.vis.update_geometry(joint)
                        for cylinder in self.left_line_set + self.right_line_set:
                            self.vis.update_geometry(cylinder)

                    self.vis.poll_events()
                    self.vis.update_renderer()

                    if first_frame:
                        view_params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
                    else:
                        self.vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)

                    self.step += 1
                    frame += 5

                    if first_frame:
                        first_frame = False

        finally:
            self.vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved frame data.")
    parser.add_argument("--directory", type=str, default="./saved_data", help="Directory with saved data")
    parser.add_argument("--table_frame", type=str, default="robot_table_pointcloud", help="Directory with saved data")

    args = parser.parse_args()

    assert os.path.exists(args.directory), f"given directory: {args.directory} not exists"
    visualizer = TableDataVisualizer(args.directory, args.table_frame)

    dir_path = args.directory

    visualizer.right_hand_offset = np.loadtxt("{}/calib_offset.txt".format(args.directory))
    visualizer.right_hand_ori_offset = np.loadtxt("{}/calib_ori_offset.txt".format(args.directory))
    visualizer.left_hand_offset = np.loadtxt("{}/calib_offset_left.txt".format(args.directory))
    visualizer.left_hand_ori_offset = np.loadtxt("{}/calib_ori_offset_left.txt".format(args.directory))

    visualizer.replay_all_frames()
