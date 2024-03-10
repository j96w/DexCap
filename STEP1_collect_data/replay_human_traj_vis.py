"""
Visualize saved pcd file and poses

Example usage:
(1) for all frames
python replay_human_traj_vis.py --directory ./saved_data/

(2) for calibration
python replay_human_traj_vis.py --directory ./saved_data/ -calib
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
from visualizer import *

def on_press(key):
    global next_frame, delta_movement_accu, delta_ori_accu, delta_movement_accu_left, delta_ori_accu_left, adjust_movement, adjust_right, frame, step

    # Determine the OS
    os_type = platform.system()

    # Common actions
    if adjust_right:
        data_to_adjust = delta_movement_accu if adjust_movement else delta_ori_accu
    else:
        data_to_adjust = delta_movement_accu_left if adjust_movement else delta_ori_accu_left

    if os_type == "Linux":
        # Linux-specific key bindings
        if key.char == '6':  # x negative
            data_to_adjust[0] += step
        elif key.char == '4':  # x positive
            data_to_adjust[0] += -step
        elif key.char == '8':  # y positive
            data_to_adjust[1] += step
        elif key.char == '2':  # y negative
            data_to_adjust[1] += -step
        elif key.char == "7":  # z positive
            data_to_adjust[2] += step
        elif key.char == "9":  # z negative
            data_to_adjust[2] += -step
        elif key.char == "3":
            delta_movement_accu *= 0.0
            delta_ori_accu *= 0.0
            delta_movement_accu_left *= 0.0
            delta_ori_accu_left *= 0.0
            next_frame = True
        elif key.char == "0":
            if (delta_movement_accu != np.array([0.0, 0.0, 0.0])).any() or (delta_ori_accu != np.array([0.0, 0.0, 0.0])).any() or (delta_movement_accu_left != np.array([0.0, 0.0, 0.0])).any() or (delta_ori_accu_left != np.array([0.0, 0.0, 0.0])).any():
                frame_dir = "./tmp_calib/"
                np.savetxt(os.path.join(frame_dir, f"frame_{frame}.txt"), delta_movement_accu)
                np.savetxt(os.path.join(frame_dir, f"frame_{frame}_ori.txt"), delta_ori_accu)
                np.savetxt(os.path.join(frame_dir, f"frame_{frame}_left.txt"), delta_movement_accu_left)
                np.savetxt(os.path.join(frame_dir, f"frame_{frame}_ori_left.txt"), delta_ori_accu_left)

            delta_movement_accu *= 0.0
            delta_ori_accu *= 0.0
            delta_movement_accu_left *= 0.0
            delta_ori_accu_left *= 0.0
            next_frame = True
        elif key == keyboard.Key.space:
            adjust_movement = not adjust_movement
        elif key == keyboard.Key.enter:
            adjust_right = not adjust_right
        else:
            print("Key error", key)

    elif os_type == "Windows":
        # Windows-specific key bindings are handled in the AttributeError section
        if key == keyboard.Key.right:  # x negative
            data_to_adjust[0] += step
        elif key == keyboard.Key.left:  # x positive
            data_to_adjust[0] += -step
        elif key == keyboard.Key.up:  # y positive
            data_to_adjust[1] += step
        elif key == keyboard.Key.down:  # y negative
            data_to_adjust[1] += -step
        elif key == keyboard.Key.home:  # z positive
            data_to_adjust[2] += step
        elif key == keyboard.Key.page_up:  # z negative
            data_to_adjust[2] += -step
        elif key == keyboard.Key.page_down:
            delta_movement_accu *= 0.0
            delta_ori_accu *= 0.0
            delta_movement_accu_left *= 0.0
            delta_ori_accu_left *= 0.0
            next_frame = True
        elif key == keyboard.Key.insert:
            if (delta_movement_accu != np.array([0.0, 0.0, 0.0])).any() or (delta_ori_accu != np.array([0.0, 0.0, 0.0])).any() or (delta_movement_accu_left != np.array([0.0, 0.0, 0.0])).any() or (delta_ori_accu_left != np.array([0.0, 0.0, 0.0])).any():
                frame_dir = "./tmp_calib/"
                np.savetxt(os.path.join(frame_dir, f"frame_{frame}.txt"), delta_movement_accu)
                np.savetxt(os.path.join(frame_dir, f"frame_{frame}_ori.txt"), delta_ori_accu)
                np.savetxt(os.path.join(frame_dir, f"frame_{frame}_left.txt"), delta_movement_accu_left)
                np.savetxt(os.path.join(frame_dir, f"frame_{frame}_ori_left.txt"), delta_ori_accu_left)

            delta_movement_accu *= 0.0
            delta_ori_accu *= 0.0
            delta_movement_accu_left *= 0.0
            delta_ori_accu_left *= 0.0
            next_frame = True
        elif key == keyboard.Key.space:
            adjust_movement = not adjust_movement
        elif key == keyboard.Key.enter:
            adjust_right = not adjust_right
        else:
            print("Key error", key)


class ReplayDataVisualizer(DataVisualizer):
    def __init__(self, directory):
        super().__init__(directory)

    def replay_keyframes_calibration(self):
        """
        Visualize a single frame
        """
        global delta_movement_accu, delta_ori_accu, next_frame, frame
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
    parser.add_argument("--default", type=str, default="default_offset", help="Directory with saved data")
    parser.add_argument("--calib", action='store_true', help="Visualize keyframe and enable calibration. If not, visualize all frames.")

    args = parser.parse_args()

    assert os.path.exists(args.directory), f"given directory: {args.directory} not exists"
    visualizer = ReplayDataVisualizer(args.directory)

    if args.calib:
        visualizer.right_hand_offset = np.loadtxt(os.path.join(args.default, "calib_offset.txt"))
        visualizer.right_hand_ori_offset = np.loadtxt(os.path.join(args.default, "calib_ori_offset.txt"))
        visualizer.left_hand_offset = np.loadtxt(os.path.join(args.default, "calib_offset_left.txt"))
        visualizer.left_hand_ori_offset = np.loadtxt(os.path.join(args.default, "calib_ori_offset_left.txt"))

        # Check if out_directory exists
        if os.path.exists("tmp_calib"):
            response = (
                input(
                    f"tmp_calib already exists. Do you want to override? (y/n): "
                )
                .strip()
                .lower()
            )
            if response != "y":
                print("Exiting program without overriding the existing directory.")
                sys.exit()
            else:
                shutil.rmtree("tmp_calib")
        os.makedirs("tmp_calib", exist_ok=True)
        visualizer.replay_keyframes_calibration()
    else:
        visualizer.right_hand_offset = np.loadtxt("{}/calib_offset.txt".format(args.directory))
        visualizer.right_hand_ori_offset = np.loadtxt("{}/calib_ori_offset.txt".format(args.directory))
        visualizer.left_hand_offset = np.loadtxt("{}/calib_offset_left.txt".format(args.directory))
        visualizer.left_hand_ori_offset = np.loadtxt("{}/calib_ori_offset_left.txt".format(args.directory))
        visualizer.replay_all_frames()
