import argparse
import os
import json
import h5py
import imageio
import sys
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2
import re
import torch
import open3d as o3d
from transforms3d.euler import quat2mat

from glob import glob
from threading import Thread, Lock
from run_trained_agent_utils import _back_project_point, apply_pose_matrix
from transforms3d.quaternions import mat2quat
from scipy.spatial.transform import Rotation
from transforms3d.axangles import axangle2mat

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS


class VisOnlyEnv:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.lock = Lock()
        self.running = False
        self.current_image = None
        self.current_index = 0

        self.agentview_image_data = None
        self.robot0_eef_pos_data = None
        self.robot0_eef_quat_data = None
        self.robot0_eef_hand_data = None
        self.robot0_eef = None

        self.action_data = None
        self.done_data = None
        self.state_data = None

        self.demo_index = 0
        self.sorted_demos = None
        self.load_hdf5_data()

    def load_hdf5_data(self):
        with h5py.File(self.hdf5_file, 'r') as file:
            data_group = file['data']
            self.sorted_demos = sorted(data_group.keys(), key=lambda x: int(x.split('_')[-1]))
            self._load_demo(self.sorted_demos[0])

    def _load_demo(self, demo_name):
        with h5py.File(self.hdf5_file, 'r') as file:
            demo_group = file['data'][demo_name]

            self.agentview_image_data = demo_group['obs']['agentview_image'][:]
            self.robot0_eef_pos_data = demo_group['obs']['robot0_eef_pos'][:]
            self.robot0_eef_quat_data = demo_group['obs']['robot0_eef_quat'][:]
            self.robot0_eef_hand_data = demo_group['obs']['robot0_eef_hand'][:]
            self.robot0_eef = np.concatenate((self.robot0_eef_pos_data, self.robot0_eef_quat_data, self.robot0_eef_hand_data), axis=-1)

            self.action_data = demo_group['actions'][:]
            self.done_data = demo_group['dones'][:]
            self.state_data = demo_group['states'][:]

            self.current_index = 0

    def get_state(self):
        with self.lock:
            if self.current_index < len(self.agentview_image_data):
                self.current_image = self.agentview_image_data[self.current_index]
                self.current_image = self._draw_action(self.current_image, self.robot0_eef[self.current_index], self.state_data[self.current_index], self.action_data[self.current_index])

                image_resized = cv2.resize(self.current_image, (84, 84))
                robot0_eef_pos = self.robot0_eef_pos_data[self.current_index]
                robot0_eef_quat = self.robot0_eef_quat_data[self.current_index]
                robot0_eef_hand_data = self.robot0_eef_hand_data[self.current_index]
                return_state = {
                    'agentview_image': image_resized,
                    'robot0_eef_pos': robot0_eef_pos,
                    'robot0_eef_quat': robot0_eef_quat,
                    'robot0_eef_hand_data': robot0_eef_hand_data,
                }

                done = bool(self.done_data[self.current_index])
                self.current_index += 1
                return return_state, done
            else:
                return None, True  # Return True for done if no more images

    def reset(self):
        with self.lock:
            self.demo_index = (self.demo_index + 1) % len(self.sorted_demos)
            self._load_demo(self.sorted_demos[self.demo_index])

    def _visualize(self):
        cv2.namedWindow('Visualization', cv2.WINDOW_NORMAL)
        while self.running:
            with self.lock:
                if self.current_image is not None:
                    cv2.imshow('Visualization', cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
                        self.running = False
            cv2.waitKey(25)  # Wait a bit before trying to get the next frame

        cv2.destroyAllWindows()

    def _draw_ee_hand(self, image, robot0_eef, corrected_pose):
        translation = robot0_eef[0:3]
        rotation_quat = robot0_eef[3:7]
        rotation_mat = quat2mat(rotation_quat)
        pose_3 = np.eye(4)
        pose_3[:3, :3] = rotation_mat
        pose_3[:3, 3] = translation
        corrected_pose = corrected_pose.reshape((4, 4))

        hand = robot0_eef[7:]
        hand = hand.reshape((21, 3))
        hand = apply_pose_matrix(hand, pose_3)

        o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280, 720,
            898.2010498046875,
            897.86669921875,
            657.4981079101562,
            364.30950927734375)

        right_hand_joint_points_homogeneous = np.hstack((hand, np.ones((hand.shape[0], 1))))
        right_hand_transformed_points_homogeneous = np.dot(right_hand_joint_points_homogeneous, np.linalg.inv(corrected_pose).T)
        right_hand_points_to_project = right_hand_transformed_points_homogeneous[:, :3] / right_hand_transformed_points_homogeneous[:, [3]]
        right_hand_back_projected_points = [_back_project_point(point, o3d_depth_intrinsic.intrinsic_matrix) for point in right_hand_points_to_project]

        for i in range(len(right_hand_back_projected_points)):
            u, v = right_hand_back_projected_points[i]
            u = int(float(u) / 1280 * 84)
            v = int(float(v) / 720 * 84)

            if i in [0, 1, 5, 9, 13, 17]:
                cv2.circle(image, (u, v), 2, (0, 255, 0), -1)
            elif i in [4, 8, 12, 16, 20]:
                cv2.circle(image, (u, v), 2, (255, 0, 0), -1)
            else:
                cv2.circle(image, (u, v), 2, (0, 0, 255), -1)
            if i in [1, 2, 3, 4]:
                cv2.circle(image, (u, v), 2, (255, 0, 255), -1)

        return image


    def _draw_action(self, image, robot0_eef, corrected_pose, action):
        translation = robot0_eef[0:3]
        rotation_quat = robot0_eef[3:7]
        rotation_mat = quat2mat(rotation_quat)
        pose_3 = np.eye(4)
        pose_3[:3, :3] = rotation_mat
        pose_3[:3, 3] = translation
        corrected_pose = corrected_pose.reshape((4, 4))

        hand_root = np.array([[0.0, 0.0, 0.0]])
        hand_root = apply_pose_matrix(hand_root, pose_3)

        action_trans = action[:3] / 10.0 * 6.0
        hand_root_next = hand_root + action_trans

        hand_root = np.concatenate((hand_root, hand_root_next), axis=0)

        o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280, 720,
            898.2010498046875,
            897.86669921875,
            657.4981079101562,
            364.30950927734375)

        hand_root_points_homogeneous = np.hstack((hand_root, np.ones((hand_root.shape[0], 1))))
        hand_root_homogeneous_transformed_points_homogeneous = np.dot(hand_root_points_homogeneous, np.linalg.inv(corrected_pose).T)
        hand_root_points_to_project = hand_root_homogeneous_transformed_points_homogeneous[:, :3] / hand_root_homogeneous_transformed_points_homogeneous[:, [3]]
        hand_root_back_projected_points = [_back_project_point(point, o3d_depth_intrinsic.intrinsic_matrix) for point in hand_root_points_to_project]

        # draw arrow first
        u0, v0 = hand_root_back_projected_points[0]
        u1, v1 = hand_root_back_projected_points[1]
        cv2.arrowedLine(image, [int(float(u0) / 1280 * 84), int(float(v0) / 720 * 84)], [int(float(u1) / 1280 * 84), int(float(v1) / 720 * 84)], (0, 0, 255), 2)

        # draw starting and ending point
        for i in range(len(hand_root_back_projected_points)):
            u, v = hand_root_back_projected_points[i]
            u = int(float(u) / 1280 * 84)
            v = int(float(v) / 720 * 84)
            if i in [0]:
                cv2.circle(image, (u, v), 2, (0, 255, 0), -1)
            elif i in [1]:
                cv2.circle(image, (u, v), 2, (255, 0, 0), -1)

        return image

    def start_visualization(self):
        self.running = True
        self.visualization_thread = Thread(target=self._visualize)
        self.visualization_thread.start()

    def stop_visualization(self):
        self.running = False
        self.visualization_thread.join()


def run_trained_agent(args):
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)

    if args.dp_eval_steps is not None:
        assert algo_name == "diffusion_policy"
        log_warning("setting @num_inference_steps to {}".format(args.dp_eval_steps))

        # HACK: modify the config, then dump to json again and write to ckpt_dict
        tmp_config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        with tmp_config.values_unlocked():
            if tmp_config.algo.ddpm.enabled:
                tmp_config.algo.ddpm.num_inference_timesteps = args.dp_eval_steps
            elif tmp_config.algo.ddim.enabled:
                tmp_config.algo.ddim.num_inference_timesteps = args.dp_eval_steps
            else:
                raise Exception("should not reach here")
        ckpt_dict['config'] = tmp_config.dump()

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    # HACK: assume absolute actions for now if using diffusion policy on real robot
    if (algo_name == "diffusion_policy") and EnvUtils.is_real_robot_gprs_env(env_meta=ckpt_dict["env_metadata"]):
        ckpt_dict["env_metadata"]["env_kwargs"]["absolute_actions"] = True

    # create environment from saved checkpoint
    env = vis_only_env(args.dataset_path)

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)



vis_env = VisOnlyEnv('/media/jeremy/cde0dfff-70f1-4c1c-82aa-e0d469c14c62/dp/mimicgen_environments/datasets/core/hand_d0.hdf5')
vis_env.start_visualization()
while True:
    state, done = vis_env.get_state()
    if done:
        vis_env.reset()
    time.sleep(0.01)

#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     # Path to trained model
#     parser.add_argument(
#         "--agent",
#         type=str,
#         required=True,
#         help="path to saved checkpoint pth file",
#     )
#
#     # If provided, an hdf5 file will be written with the rollout data
#     parser.add_argument(
#         "--dataset_path",
#         type=str,
#         default=None,
#         help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
#     )
#
#     # for seeding before starting rollouts
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=None,
#         help="(optional) set seed for rollouts",
#     )
#
#     args = parser.parse_args()
#     res_str = None
#     try:
#         run_trained_agent(args)
#     except Exception as e:
#         res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
#         if args.error_path is not None:
#             # write traceback to file
#             f = open(args.error_path, "w")
#             f.write(res_str)
#             f.close()
#         raise e