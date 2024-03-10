"""
Example usage

python data_recording.py -s --store_hand -o ./save_data_scenario_1
"""

import argparse
import copy
import numpy as np
import open3d as o3d
import os
import shutil
import sys
import pyrealsense2 as rs
import cv2

from enum import IntEnum
from realsense_helper import get_profiles
from transforms3d.quaternions import axangle2quat, qmult, quat2mat, mat2quat
import redis
import concurrent.futures
from hyperparameters import *


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def save_frame(
    frame_id,
    out_directory,
    color_buffer,
    depth_buffer,
    pose_buffer,
    pose2_buffer,
    pose3_buffer,
    rightHandJoint_buffer,
    leftHandJoint_buffer,
    rightHandJointOri_buffer,
    leftHandJointOri_buffer,
    save_hand,
):
    frame_directory = os.path.join(out_directory, f"frame_{frame_id}")
    os.makedirs(frame_directory, exist_ok=True)

    cv2.imwrite(
        os.path.join(frame_directory, "color_image.jpg"),
        color_buffer[frame_id][:, :, ::-1],
    )
    cv2.imwrite(
        os.path.join(frame_directory, "depth_image.png"), depth_buffer[frame_id]
    )

    np.savetxt(os.path.join(frame_directory, "pose.txt"), pose_buffer[frame_id])
    np.savetxt(os.path.join(frame_directory, "pose_2.txt"), pose2_buffer[frame_id])
    np.savetxt(os.path.join(frame_directory, "pose_3.txt"), pose3_buffer[frame_id])

    if save_hand:
        np.savetxt(
            os.path.join(frame_directory, "right_hand_joint.txt"),
            rightHandJoint_buffer[frame_id],
        )
        np.savetxt(
            os.path.join(frame_directory, "left_hand_joint.txt"),
            leftHandJoint_buffer[frame_id],
        )
        np.savetxt(
            os.path.join(frame_directory, "right_hand_joint_ori.txt"),
            rightHandJointOri_buffer[frame_id],
        )
        np.savetxt(
            os.path.join(frame_directory, "left_hand_joint_ori.txt"),
            leftHandJointOri_buffer[frame_id],
        )

    return f"frame {frame_id + 1} saved"


class RealsesneProcessor:
    def __init__(
        self,
        first_t265_serial,
        second_t265_serial,
        thrid_t265_serial,
        total_frame,
        store_frame=False,
        out_directory=None,
        save_hand=False,
        enable_visualization=True,
    ):
        self.first_t265_serial = first_t265_serial
        self.second_t265_serial = second_t265_serial
        self.thrid_t265_serial = thrid_t265_serial
        self.store_frame = store_frame
        self.out_directory = out_directory
        self.total_frame = total_frame
        self.save_hand = save_hand
        self.enable_visualization = enable_visualization
        self.rds = None

        self.color_buffer = []
        self.depth_buffer = []

        self.pose_buffer = []
        self.pose2_buffer = []
        self.pose3_buffer = []

        self.pose2_image_buffer = []
        self.pose3_image_buffer = []

        self.rightHandJoint_buffer = []
        self.leftHandJoint_buffer = []
        self.rightHandJointOri_buffer = []
        self.leftHandJointOri_buffer = []

    def get_rs_t265_config(self, t265_serial, t265_pipeline):
        t265_config = rs.config()
        t265_config.enable_device(t265_serial)
        t265_config.enable_stream(rs.stream.pose)

        return t265_config

    def configure_stream(self):
        # connect to redis server
        if self.save_hand:
            self.rds = redis.Redis(host="localhost", port=6669, db=0)

        # Create a pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        color_profiles, depth_profiles = get_profiles()
        w, h, fps, fmt = depth_profiles[1]
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        w, h, fps, fmt = color_profiles[18]
        config.enable_stream(rs.stream.color, w, h, fmt, fps)

        # Configure the t265 1 stream
        ctx = rs.context()
        self.t265_pipeline = rs.pipeline(ctx)
        t265_config = rs.config()
        t265_config.enable_device(self.first_t265_serial)

        # Configure the t265 2 stream
        ctx_2 = rs.context()
        self.t265_pipeline_2 = rs.pipeline(ctx_2)
        t265_config_2 = self.get_rs_t265_config(
            self.second_t265_serial, self.t265_pipeline_2
        )

        # Configure the t265 3 stream
        ctx_3 = rs.context()
        self.t265_pipeline_3 = rs.pipeline(ctx_3)
        t265_config_3 = self.get_rs_t265_config(
            self.thrid_t265_serial, self.t265_pipeline_3
        )

        self.t265_pipeline.start(t265_config)
        self.t265_pipeline_2.start(t265_config_2)
        self.t265_pipeline_3.start(t265_config_3)

        pipeline_profile = self.pipeline.start(config)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
        self.depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.vis = None
        if self.enable_visualization:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.get_view_control().change_field_of_view(step=1.0)

    def get_rgbd_frame_from_realsense(self, enable_visualization=False):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = (
            np.asanyarray(aligned_depth_frame.get_data()) // 4
        )  # L515 camera need to divide by 4 to get metric in meter
        color_image = np.asanyarray(color_frame.get_data())

        rgbd = None
        if enable_visualization:
            depth_image_o3d = o3d.geometry.Image(depth_image)
            color_image_o3d = o3d.geometry.Image(color_image)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image_o3d,
                depth_image_o3d,
                depth_trunc=4.0,
                convert_rgb_to_intensity=False,
            )
        return rgbd, depth_image, color_image

    @staticmethod
    def frame_to_pose_conversion(input_t265_frames):
        pose_frame = input_t265_frames.get_pose_frame()
        pose_data = pose_frame.get_pose_data()
        pose_3x3 = quat2mat(
            np.array(
                [
                    pose_data.rotation.w,
                    pose_data.rotation.x,
                    pose_data.rotation.y,
                    pose_data.rotation.z,
                ]
            )
        )
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = pose_3x3
        pose_4x4[:3, 3] = [
            pose_data.translation.x,
            pose_data.translation.y,
            pose_data.translation.z,
        ]
        return pose_4x4

    def process_frame(self):
        frame_count = 0
        first_frame = True

        try:
            while frame_count < self.total_frame:
                t265_frames = self.t265_pipeline.wait_for_frames()
                t265_frames_2 = self.t265_pipeline_2.wait_for_frames()
                t265_frames_3 = self.t265_pipeline_3.wait_for_frames()
                rgbd, depth_frame, color_frame = self.get_rgbd_frame_from_realsense()

                # get pose data for t265 1
                pose_4x4 = RealsesneProcessor.frame_to_pose_conversion(
                    input_t265_frames=t265_frames
                )
                pose_4x4_2 = RealsesneProcessor.frame_to_pose_conversion(
                    input_t265_frames=t265_frames_2
                )
                pose_4x4_3 = RealsesneProcessor.frame_to_pose_conversion(
                    input_t265_frames=t265_frames_3
                )

                if self.save_hand:
                    # get hand joint data
                    leftHandJointXyz = np.frombuffer(
                        self.rds.get("rawLeftHandJointXyz"), dtype=np.float64
                    ).reshape(21, 3)
                    rightHandJointXyz = np.frombuffer(
                        self.rds.get("rawRightHandJointXyz"), dtype=np.float64
                    ).reshape(21, 3)
                    leftHandJointOrientation = np.frombuffer(
                        self.rds.get("rawLeftHandJointOrientation"), dtype=np.float64
                    ).reshape(21, 4)
                    rightHandJointOrientation = np.frombuffer(
                        self.rds.get("rawRightHandJointOrientation"), dtype=np.float64
                    ).reshape(21, 4)

                corrected_pose = pose_4x4 @ between_cam

                # Convert to Open3D format L515
                o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    1280,
                    720,
                    898.2010498046875,
                    897.86669921875,
                    657.4981079101562,
                    364.30950927734375,
                )

                if first_frame:
                    if self.enable_visualization:
                        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd, o3d_depth_intrinsic
                        )
                        pcd.transform(corrected_pose)

                        rgbd_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.3
                        )
                        rgbd_mesh.transform(corrected_pose)
                        rgbd_previous_pose = copy.deepcopy(corrected_pose)

                        chest_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.3
                        )
                        chest_mesh.transform(pose_4x4)
                        chest_previous_pose = copy.deepcopy(pose_4x4)

                        left_hand_mesh = (
                            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                        )
                        left_hand_mesh.transform(pose_4x4_2)
                        left_hand_previous_pose = copy.deepcopy(pose_4x4_2)

                        right_hand_mesh = (
                            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                        )
                        right_hand_mesh.transform(pose_4x4_3)
                        right_hand_previous_pose = copy.deepcopy(pose_4x4_3)

                        self.vis.add_geometry(pcd)
                        self.vis.add_geometry(rgbd_mesh)
                        self.vis.add_geometry(chest_mesh)
                        self.vis.add_geometry(left_hand_mesh)
                        self.vis.add_geometry(right_hand_mesh)

                        view_params = (
                            self.vis.get_view_control().convert_to_pinhole_camera_parameters()
                        )
                    first_frame = False
                else:
                    if self.enable_visualization:
                        new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd, o3d_depth_intrinsic
                        )
                        new_pcd.transform(corrected_pose)

                        rgbd_mesh.transform(np.linalg.inv(rgbd_previous_pose))
                        rgbd_mesh.transform(corrected_pose)
                        rgbd_previous_pose = copy.deepcopy(corrected_pose)

                        chest_mesh.transform(np.linalg.inv(chest_previous_pose))
                        chest_mesh.transform(pose_4x4)
                        chest_previous_pose = copy.deepcopy(pose_4x4)

                        left_hand_mesh.transform(np.linalg.inv(left_hand_previous_pose))
                        left_hand_mesh.transform(pose_4x4_2)
                        left_hand_previous_pose = copy.deepcopy(pose_4x4_2)

                        right_hand_mesh.transform(
                            np.linalg.inv(right_hand_previous_pose)
                        )
                        right_hand_mesh.transform(pose_4x4_3)
                        right_hand_previous_pose = copy.deepcopy(pose_4x4_3)

                        pcd.points = new_pcd.points
                        pcd.colors = new_pcd.colors

                        self.vis.update_geometry(pcd)
                        self.vis.update_geometry(rgbd_mesh)
                        self.vis.update_geometry(chest_mesh)
                        self.vis.update_geometry(left_hand_mesh)
                        self.vis.update_geometry(right_hand_mesh)

                        self.vis.get_view_control().convert_from_pinhole_camera_parameters(
                            view_params
                        )

                if self.enable_visualization:
                    self.vis.poll_events()
                    self.vis.update_renderer()

                if self.store_frame:
                    self.depth_buffer.append(copy.deepcopy(depth_frame))
                    self.color_buffer.append(copy.deepcopy(color_frame))

                    self.pose_buffer.append(copy.deepcopy(pose_4x4))
                    self.pose2_buffer.append(copy.deepcopy(pose_4x4_2))
                    self.pose3_buffer.append(copy.deepcopy(pose_4x4_3))

                    if self.save_hand:
                        self.rightHandJoint_buffer.append(
                            copy.deepcopy(rightHandJointXyz)
                        )
                        self.leftHandJoint_buffer.append(
                            copy.deepcopy(leftHandJointXyz)
                        )
                        self.rightHandJointOri_buffer.append(
                            copy.deepcopy(rightHandJointOrientation)
                        )
                        self.leftHandJointOri_buffer.append(
                            copy.deepcopy(leftHandJointOrientation)
                        )

                frame_count += 1
                print("streamed frame {}".format(frame_count))
        except Exception as e:
            print("An error occurred:", e)
        finally:
            self.t265_pipeline.stop()
            self.t265_pipeline_2.stop()
            self.t265_pipeline_3.stop()
            self.pipeline.stop()
            if self.enable_visualization:
                self.vis.destroy_window()

            if self.store_frame:
                print("saving frames...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            save_frame,
                            frame_id,
                            self.out_directory,
                            self.color_buffer,
                            self.depth_buffer,
                            self.pose_buffer,
                            self.pose2_buffer,
                            self.pose3_buffer,
                            self.rightHandJoint_buffer,
                            self.leftHandJoint_buffer,
                            self.rightHandJointOri_buffer,
                            self.leftHandJointOri_buffer,
                            self.save_hand,
                        )
                        for frame_id in range(frame_count)
                    ]

                    for future in concurrent.futures.as_completed(futures):
                        print(future.result(), f" total frame: {frame_count}")


import concurrent.futures


def main(args):
    realsense_processor = RealsesneProcessor(
        first_t265_serial="11622110012",
        second_t265_serial="909212110944",
        thrid_t265_serial="929122111181",
        total_frame=10000,
        store_frame=args.store_frame,
        out_directory=args.out_directory,
        save_hand=args.store_hand,
        enable_visualization=args.enable_vis,
    )
    realsense_processor.configure_stream()
    realsense_processor.process_frame()


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process frames and save data.")
    parser.add_argument(
        "-s",
        "--store_frame",
        action="store_true",
        help="Flag to indicate whether to store frames",
    )
    parser.add_argument(
        "--store_hand",
        action="store_true",
        help="Flag to indicate whether to store hand joint position and orientation",
    )
    parser.add_argument(
        "-v",
        "--enable_vis",
        action="store_true",
        help="Flag to indicate whether to enable open3d visualization",
    )
    parser.add_argument(
        "-o",
        "--out_directory",
        type=str,
        help="Output directory for saved data",
        default="./saved_data",
    )

    args = parser.parse_args()

    # Check if out_directory exists
    if os.path.exists(args.out_directory):
        response = (
            input(
                f"{args.out_directory} already exists. Do you want to override? (y/n): "
            )
            .strip()
            .lower()
        )
        if response != "y":
            print("Exiting program without overriding the existing directory.")
            sys.exit()
        else:
            shutil.rmtree(args.out_directory)
    if args.store_frame:
        os.makedirs(args.out_directory, exist_ok=True)

    # If user chooses to override, remove the existing directory
    main(args)
