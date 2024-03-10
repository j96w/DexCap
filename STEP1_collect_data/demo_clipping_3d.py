"""
Visualize pcd file and select the start and end frame of each demo

Example usage:
(1) for all frames
python demo_clipping_3d.py --directory ./saved_data/
"""

import argparse
import os
import copy
import zmq
import cv2
import sys
import json
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

clip_marks = []
current_clip = {}
next_frame = False
previous_frame = False

def on_press(key):
    global next_frame, previous_frame, delta_movement_accu, delta_ori_accu, delta_movement_accu_left, delta_ori_accu_left, adjust_movement, adjust_right, frame, step, dataset_folder, clip_marks, current_clip

    # Determine the OS
    os_type = platform.system()
    assert os_type == "Windows"

    frame_folder = 'frame_{}'.format(frame)
    # Windows-specific key bindings are handled in the AttributeError section
    if key == keyboard.Key.up:  # y positive
        with open(os.path.join(dataset_folder, 'clip_marks.json'), 'w') as f:
            json.dump(clip_marks, f, indent=4)
    elif key == keyboard.Key.down:  # y negative
        previous_frame = True
    elif key == keyboard.Key.page_down:
        next_frame = True
    elif key == keyboard.Key.end:
        if 'start' in current_clip.keys():
            print("end", frame_folder)
            current_clip['end'] = frame_folder
            clip_marks.append(current_clip)
            current_clip = {}
    elif key == keyboard.Key.insert:
        print("start", frame_folder)
        current_clip['start'] = frame_folder
    else:
        print("Key error", key)


class ReplayDataVisualizer(DataVisualizer):
    def __init__(self, directory):
        super().__init__(directory)

    def replay_frames(self):
        """
        Visualize a single frame
        """
        global delta_movement_accu, delta_ori_accu, next_frame, previous_frame, frame
        if self.R_delta_init is None:
            self.initialize_canonical_frame()

        self._load_frame_data(frame)

        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.coord_frame_1)
        self.vis.add_geometry(self.coord_frame_2)
        self.vis.add_geometry(self.coord_frame_3)
        for joint in self.left_joints + self.right_joints:
            self.vis.add_geometry(joint)
        for cylinder in self.left_line_set + self.right_line_set:
            self.vis.add_geometry(cylinder)

        next_frame = True
        try:
            with keyboard.Listener(on_press=on_press) as listener:
                while True:
                    if next_frame == True:
                        next_frame = False
                        frame += 10
                    if previous_frame == True:
                        previous_frame = False
                        frame -= 10
                    self._load_frame_data(frame)

                    self.step += 1

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
                listener.join()
        finally:
            print("cumulative_correction ", self.cumulative_correction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved frame data.")
    parser.add_argument("--directory", type=str, default="./saved_data", help="Directory with saved data")

    args = parser.parse_args()

    assert os.path.exists(args.directory), f"given directory: {args.directory} not exists"

    if os.path.exists(os.path.join(args.directory, 'clip_marks.json')):
        response = (
            input(
                f"clip_marks.json already exists. Do you want to override? (y/n): "
            )
            .strip()
            .lower()
        )
        if response != "y":
            print("Exiting program without overriding the existing directory.")
            sys.exit()

    dataset_folder = args.directory
    visualizer = ReplayDataVisualizer(args.directory)

    visualizer.right_hand_offset = np.loadtxt("{}/calib_offset.txt".format(args.directory))
    visualizer.right_hand_ori_offset = np.loadtxt("{}/calib_ori_offset.txt".format(args.directory))
    visualizer.left_hand_offset = np.loadtxt("{}/calib_offset_left.txt".format(args.directory))
    visualizer.left_hand_ori_offset = np.loadtxt("{}/calib_ori_offset_left.txt".format(args.directory))
    visualizer.replay_frames()
