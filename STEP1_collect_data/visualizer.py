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

lines = np.array([
    # Thumb
    [1, 2], [2, 3], [3, 4],
    # Index
    [5, 6], [6, 7], [7, 8],
    # Middle
    [9, 10], [10, 11], [11, 12],
    # Ring
    [13, 14], [14, 15], [15, 16],
    # Little
    [17, 18], [18, 19], [19, 20],
    # Connections between proximals
    [1, 5], [5, 9], [9, 13], [13, 17],
    # connect palm
    [0, 1], [17, 0]
])

delta_movement_accu = np.array([0.0, 0.0, 0.0])
delta_ori_accu = np.array([0.0, 0.0, 0.0])
delta_movement_accu_left = np.array([0.0, 0.0, 0.0])
delta_ori_accu_left = np.array([0.0, 0.0, 0.0])
adjust_movement = True
adjust_right = True
next_frame = False
frame = 0
step = 0.01 # tune this step size
fixed_transform = np.array([0.0, 0.0, 0.0])


def translate_wrist_to_origin(joint_positions):
    wrist_position = joint_positions[0]
    updated_positions = joint_positions - wrist_position
    return updated_positions

def apply_pose_matrix(joint_positions, pose_matrix):
    homogeneous_joint_positions = np.hstack([joint_positions, np.ones((joint_positions.shape[0], 1))])
    transformed_positions = np.dot(homogeneous_joint_positions, pose_matrix.T)
    transformed_positions_3d = transformed_positions[:, :3]
    return transformed_positions_3d

def create_or_update_cylinder(start, end, radius=0.003, cylinder_list=None, cylinder_idx=-1):
    # Calculate the length of the cylinder
    cyl_length = np.linalg.norm(end - start)

    # Create a new cylinder
    new_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=cyl_length, resolution=20, split=4)
    new_cylinder.paint_uniform_color([1, 0, 0])

    new_cylinder.translate(np.array([0, 0, cyl_length / 2]))

    # Compute the direction vector of the line segment and normalize it
    direction = end - start
    direction /= np.linalg.norm(direction)

    # Compute the rotation axis and angle
    up = np.array([0, 0, 1])  # Default up vector of cylinder
    rotation_axis = np.cross(up, direction)
    rotation_angle = np.arccos(np.dot(up, direction))

    # Compute the rotation matrix
    if np.linalg.norm(rotation_axis) != 0:
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        new_cylinder.rotate(rotation_matrix, center=np.array([0, 0, 0]))

    # Translate the new cylinder to the start position
    new_cylinder.translate(start)

    # Copy new cylinder to the original one if it exists
    if cylinder_list[cylinder_idx] is not None:
        cylinder_list[cylinder_idx].vertices = new_cylinder.vertices
        cylinder_list[cylinder_idx].triangles = new_cylinder.triangles
        cylinder_list[cylinder_idx].vertex_normals = new_cylinder.vertex_normals
        cylinder_list[cylinder_idx].vertex_colors = new_cylinder.vertex_colors
    else:
        cylinder_list[cylinder_idx] = new_cylinder

class DataVisualizer:
    def __init__(self, directory):
        self.directory = directory
        self.base_pcd = None
        self.pcd = None
        self.img_backproj = None
        self.coord_frame_1 = None
        self.coord_frame_2 = None
        self.coord_frame_3 = None

        self.right_hand_offset = None
        self.right_hand_ori_offset = None
        self.left_hand_offset = None
        self.left_hand_ori_offset = None

        self.pose1_prev = np.eye(4)
        self.pose2_prev = np.eye(4)
        self.pose3_prev = np.eye(4)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_view_control().change_field_of_view(step=1.0)

        self.between_cam = np.eye(4)
        self.between_cam[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0],
                                             [0.0, 0.0, -1.0]])
        self.between_cam[:3, 3] = np.array([0.0, 0.076, 0.0])  # was 0.076

        self.between_cam_2 = np.eye(4)
        self.between_cam_2[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]])
        self.between_cam_2[:3, 3] = np.array([0.0, -0.032, 0.0])

        self.between_cam_3 = np.eye(4)
        self.between_cam_3[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]])
        self.between_cam_3[:3, 3] = np.array([0.0, -0.064, 0.0])

        self.canonical_t265_ori = None
        # visualize left hand 21 joint
        self.left_joints = []
        self.right_joints = []
        self.left_line_set = [None for _ in lines]
        self.right_line_set = [None for _ in lines]
        for i in range(21):
            for joint in [self.left_joints, self.right_joints]:
                # larger sphere for tips and wrist, and smaller for other joints
                radius = 0.011 if i in [0, 4, 8, 12, 16, 20] else 0.007
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                # color wrist and proximal joint as green
                if i in [0, 1, 5, 9, 13, 17]:
                    sphere.paint_uniform_color([0, 1, 0])
                # color tip as red
                elif i in [4, 8, 12, 16, 20]:
                    sphere.paint_uniform_color([1, 0, 0])
                # color other joint as blue
                else:
                    sphere.paint_uniform_color([0, 0, 1])
                    # color thumb as pink
                if i in [1, 2, 3, 4]:
                    sphere.paint_uniform_color([1, 0, 1])
                joint.append(sphere)
                self.vis.add_geometry(sphere)

        self.step = 0
        self.distance_buffer = []
        self.R_delta_init = None

        self.cumulative_correction = np.array([0.0, 0.0, 0.0])

    def initialize_canonical_frame(self):
        if self.R_delta_init is None:
            self._load_frame_data(0)
            pose_ori_matirx = self.pose3_prev[:3, :3]
            pose_ori_correction_matrix = np.dot(np.array([[0, -1, 0],
                                                          [0, 0, 1],
                                                          [1, 0, 0]]), euler2mat(0, 0, 0))
            pose_ori_matirx = np.dot(pose_ori_matirx, pose_ori_correction_matrix)

            self.canonical_t265_ori = np.array([[1, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, -1]])
            x_angle, y_angle, z_angle = mat2euler(self.pose3_prev[:3, :3])
            self.canonical_t265_ori = np.dot(self.canonical_t265_ori, euler2mat(-z_angle, x_angle + 0.3, y_angle))

            self.R_delta_init = np.dot(self.canonical_t265_ori, pose_ori_matirx.T)

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

    def replay_all_frames(self):
        """
        Visualize all frames continuously
        """
        try:
            if self.R_delta_init is None:
                self.initialize_canonical_frame()

            frame = 0
            first_frame = True
            while True:
                if not self._load_frame_data(frame):
                    break

                if first_frame:
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

    def _back_project_point(self, point, intrinsics):
        """ Back-project a single point from 3D to 2D image space """
        x, y, z = point
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        u = (x * fx / z) + cx
        v = (y * fy / z) + cy

        return int(u), int(v)

    def _load_frame_data(self, frame, vis_2d=False, load_table_points=False):
        """
        Load point cloud and poses for a given frame

        @param frame: frame count in integer
        @return whether we can successfully load all data from frame subdirectory
        """
        global delta_movement_accu, delta_ori_accu, delta_movement_accu_left, delta_ori_accu_left
        print(f"frame {frame}")

        if adjust_movement:
            print("adjusting translation")
        else:
            print("adjusting rotation")

        # L515:
        o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280, 720,
            898.2010498046875,
            897.86669921875,
            657.4981079101562,
            364.30950927734375)

        if load_table_points: # process table frame
            table_color_image_o3d = o3d.io.read_image(os.path.join(self.table_frame, "frame_0", "color_image.jpg"))
            table_depth_image_o3d = o3d.io.read_image(os.path.join(self.table_frame, "frame_0", "depth_image.png"))
            table_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(table_color_image_o3d, table_depth_image_o3d, depth_trunc=4.0,
                                                                    convert_rgb_to_intensity=False)
            table_pose_4x4 = np.loadtxt(os.path.join(self.table_frame, "frame_0", "pose.txt"))
            table_corrected_pose = table_pose_4x4 @ self.between_cam
            self.table_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(table_rgbd, o3d_depth_intrinsic)
            self.table_pcd.transform(table_corrected_pose)

        frame_dir = os.path.join(self.directory, f"frame_{frame}")

        color_image_o3d = o3d.io.read_image(os.path.join(frame_dir, "color_image.jpg"))
        depth_image_o3d = o3d.io.read_image(os.path.join(frame_dir, "depth_image.png"))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image_o3d, depth_trunc=4.0,
                                                                  convert_rgb_to_intensity=False)

        pose_4x4 = np.loadtxt(os.path.join(frame_dir, "pose.txt"))
        if load_table_points:
            pose_4x4[:3, 3] += fixed_transform.T
        corrected_pose = pose_4x4 @ self.between_cam

        pose_path = os.path.join(frame_dir, "pose.txt")
        pose_2_path = os.path.join(frame_dir, "pose_2.txt")
        pose_3_path = os.path.join(frame_dir, "pose_3.txt")

        if not all(os.path.exists(path) for path in [pose_path, pose_2_path, pose_3_path]):
            return False

        if self.pcd is None:
            self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_depth_intrinsic)
            self.pcd.transform(corrected_pose)
        else:
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_depth_intrinsic)
            new_pcd.transform(corrected_pose)
            self.pcd.points = new_pcd.points
            self.pcd.colors = new_pcd.colors

        pose_1 = np.loadtxt(pose_path)
        if load_table_points:
            pose_1[:3, 3] += fixed_transform.T
        pose_1 = pose_1 @ self.between_cam
        pose_2 = np.loadtxt(pose_2_path)
        if load_table_points:
            pose_2[:3, 3] += fixed_transform.T
        pose_2 = pose_2 @ self.between_cam_2
        pose_3 = np.loadtxt(pose_3_path)
        if load_table_points:
            pose_3[:3, 3] += fixed_transform.T
        pose_3 = pose_3 @ self.between_cam_3

        if self.coord_frame_1 is None:
            self.coord_frame_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.coord_frame_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.coord_frame_3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        self.coord_frame_1 = self.coord_frame_1.transform(np.linalg.inv(self.pose1_prev))
        self.coord_frame_1 = self.coord_frame_1.transform(pose_1)
        self.pose1_prev = copy.deepcopy(pose_1)

        self.coord_frame_2 = self.coord_frame_2.transform(np.linalg.inv(self.pose2_prev))
        self.coord_frame_2 = self.coord_frame_2.transform(pose_2)
        self.pose2_prev = copy.deepcopy(pose_2)

        self.coord_frame_3 = self.coord_frame_3.transform(np.linalg.inv(self.pose3_prev))
        self.coord_frame_3 = self.coord_frame_3.transform(pose_3)
        self.pose3_prev = copy.deepcopy(pose_3)

        # left hand, read from joint
        left_hand_joint_xyz = np.loadtxt(os.path.join(frame_dir, "left_hand_joint.txt"))
        self.left_hand_joint_xyz = left_hand_joint_xyz
        left_hand_joint_xyz = translate_wrist_to_origin(left_hand_joint_xyz)  # canonical view
        left_hand_joint_ori = np.loadtxt(os.path.join(frame_dir, "left_hand_joint_ori.txt"))[0]
        self.left_hand_wrist_ori = left_hand_joint_ori
        left_rotation_matrix = Rotation.from_quat(left_hand_joint_ori).as_matrix().T
        left_hand_joint_xyz_reshaped = left_hand_joint_xyz[:, :, np.newaxis]
        left_transformed_joint_xyz = np.matmul(left_rotation_matrix, left_hand_joint_xyz_reshaped)
        left_hand_joint_xyz = left_transformed_joint_xyz[:, :, 0]
        left_hand_joint_xyz[:, -1] = -left_hand_joint_xyz[:, -1]  # z-axis revert
        rotation_matrix = axangle2mat(np.array([0, 1, 0]), -np.pi * 1 / 2)  # y-axis rotate
        left_hand_joint_xyz = np.dot(left_hand_joint_xyz, rotation_matrix.T)
        rotation_matrix = axangle2mat(np.array([1, 0, 0]), np.pi * 1 / 2)  # x-axis rotate
        left_hand_joint_xyz = np.dot(left_hand_joint_xyz, rotation_matrix.T)
        rotation_matrix = axangle2mat(np.array([0, 0, 1]), -np.pi * 1 / 2)  # z-axis rotate
        left_hand_joint_xyz = np.dot(left_hand_joint_xyz, rotation_matrix.T)
        left_hand_joint_xyz = np.dot(left_hand_joint_xyz, euler2mat(*self.left_hand_ori_offset).T)  # rotation calibration
        left_hand_joint_xyz = np.dot(left_hand_joint_xyz, euler2mat(*delta_ori_accu_left).T) # rotation calibration
        left_hand_joint_xyz += self.left_hand_offset # translation calibration
        left_hand_joint_xyz += delta_movement_accu_left
        left_hand_joint_xyz = apply_pose_matrix(left_hand_joint_xyz, pose_2)

        # set joint sphere and lines
        for i, sphere in enumerate(self.left_joints):
            transformation = np.eye(4)
            transformation[:3, 3] = left_hand_joint_xyz[i] - sphere.get_center()
            sphere.transform(transformation)
        for i, (x, y) in enumerate(lines):
            start = self.left_joints[x].get_center()
            start = self.left_joints[x].get_center()
            end = self.left_joints[y].get_center()
            create_or_update_cylinder(start, end, cylinder_list=self.left_line_set, cylinder_idx=i)

        # right hand, read from joint
        right_hand_joint_xyz = np.loadtxt(os.path.join(frame_dir, "right_hand_joint.txt"))
        self.right_hand_joint_xyz = right_hand_joint_xyz
        right_hand_joint_xyz = translate_wrist_to_origin(right_hand_joint_xyz)  # canonical view by translate to origin
        right_hand_joint_ori = np.loadtxt(os.path.join(frame_dir, "right_hand_joint_ori.txt"))[0]
        self.right_hand_wrist_ori = right_hand_joint_ori
        right_rotation_matrix = Rotation.from_quat(right_hand_joint_ori).as_matrix().T
        right_joint_xyz_reshaped = right_hand_joint_xyz[:, :, np.newaxis]
        right_transformed_joint_xyz = np.matmul(right_rotation_matrix, right_joint_xyz_reshaped)
        right_hand_joint_xyz = right_transformed_joint_xyz[:, :, 0]
        right_hand_joint_xyz[:, -1] = -right_hand_joint_xyz[:, -1]  # z-axis revert
        rotation_matrix = axangle2mat(np.array([0, 1, 0]), -np.pi * 1 / 2)  # y-axis rotate
        right_hand_joint_xyz = np.dot(right_hand_joint_xyz, rotation_matrix.T)
        rotation_matrix = axangle2mat(np.array([1, 0, 0]), np.pi * 1 / 2)  # x-axis rotate
        right_hand_joint_xyz = np.dot(right_hand_joint_xyz, rotation_matrix.T)
        rotation_matrix = axangle2mat(np.array([0, 0, 1]), -np.pi * 1 / 2)  # z-axis rotate
        right_hand_joint_xyz = np.dot(right_hand_joint_xyz, rotation_matrix.T)
        right_hand_joint_xyz = np.dot(right_hand_joint_xyz, euler2mat(*self.right_hand_ori_offset).T)  # rotation calibration
        right_hand_joint_xyz = np.dot(right_hand_joint_xyz, euler2mat(*delta_ori_accu).T) # rotation calibration
        right_hand_joint_xyz += self.right_hand_offset # translation calibration
        right_hand_joint_xyz += delta_movement_accu
        right_hand_joint_xyz = apply_pose_matrix(right_hand_joint_xyz, pose_3)

        if vis_2d:
            color_image = np.asarray(rgbd.color)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            left_hand_joint_points_homogeneous = np.hstack((left_hand_joint_xyz, np.ones((left_hand_joint_xyz.shape[0], 1))))
            left_hand_transformed_points_homogeneous = np.dot(left_hand_joint_points_homogeneous, np.linalg.inv(corrected_pose).T)
            left_hand_points_to_project = left_hand_transformed_points_homogeneous[:, :3] / left_hand_transformed_points_homogeneous[:, [3]]
            left_hand_back_projected_points = [self._back_project_point(point, o3d_depth_intrinsic.intrinsic_matrix) for point in left_hand_points_to_project]

            for i in range(len(left_hand_back_projected_points)):
                u, v = left_hand_back_projected_points[i]
                if i in [0, 1, 5, 9, 13, 17]:
                    cv2.circle(color_image, (u, v), 10, (0, 255, 0), -1)
                elif i in [4, 8, 12, 16, 20]:
                    cv2.circle(color_image, (u, v), 10, (255, 0, 0), -1)
                else:
                    cv2.circle(color_image, (u, v), 10, (0, 0, 255), -1)
                if i in [1, 2, 3, 4]:
                    cv2.circle(color_image, (u, v), 10, (255, 0, 255), -1)

            right_hand_joint_points_homogeneous = np.hstack((right_hand_joint_xyz, np.ones((right_hand_joint_xyz.shape[0], 1))))
            right_hand_transformed_points_homogeneous = np.dot(right_hand_joint_points_homogeneous, np.linalg.inv(corrected_pose).T)
            right_hand_points_to_project = right_hand_transformed_points_homogeneous[:, :3] / right_hand_transformed_points_homogeneous[:, [3]]
            right_hand_back_projected_points = [self._back_project_point(point, o3d_depth_intrinsic.intrinsic_matrix) for point in right_hand_points_to_project]

            for i in range(len(right_hand_back_projected_points)):
                u, v = right_hand_back_projected_points[i]
                if i in [0, 1, 5, 9, 13, 17]:
                    cv2.circle(color_image, (u, v), 10, (0, 255, 0), -1)
                elif i in [4, 8, 12, 16, 20]:
                    cv2.circle(color_image, (u, v), 10, (255, 0, 0), -1)
                else:
                    cv2.circle(color_image, (u, v), 10, (0, 0, 255), -1)
                if i in [1, 2, 3, 4]:
                    cv2.circle(color_image, (u, v), 10, (255, 0, 255), -1)

            cv2.imshow("Back-projected Points on Image", color_image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # set joint sphere and lines
        for i, sphere in enumerate(self.right_joints):
            transformation = np.eye(4)
            transformation[:3, 3] = right_hand_joint_xyz[i] - sphere.get_center()
            sphere.transform(transformation)
        for i, (x, y) in enumerate(lines):
            start = self.right_joints[x].get_center()
            end = self.right_joints[y].get_center()
            create_or_update_cylinder(start, end, cylinder_list=self.right_line_set, cylinder_idx=i)

        return True