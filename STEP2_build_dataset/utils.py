import os
import copy
import yaml
import cv2
import numpy as np
import open3d as o3d
import pybullet as p
from transforms3d.quaternions import axangle2quat, qmult
from transforms3d.quaternions import mat2quat
from scipy.spatial.transform import Rotation
from transforms3d.axangles import axangle2mat
from transforms3d.euler import mat2euler, quat2mat, euler2mat
from hyperparameters import *

def extract_dataset_folder_last_two_digits(dir_name):
    # Extract the last two characters, ensure they are digits, and convert to integer
    last_two = dir_name[-2:]  # Get the last two characters
    if last_two.isdigit():
        return int(last_two)
    else:
        return -1  # Return -1 (or some other value) if there are no digits

def resize_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (84, 84))
    return image_resized

def _back_project_batch(points, intrinsics):
    """ Back-project a batch of points from 3D to 2D image space using vectorized operations """
    points = np.array(points)

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    projected_points = np.vstack((u, v)).T.astype(int)

    return projected_points

def transform_right_leap_pointcloud_to_camera_frame(hand_points, pose_data):
    pose_3 = np.eye(4)
    pose_3[:3, :3] = quat2mat(pose_data[3:7])
    pose_3[:3, 3] = pose_data[:3]
    init_mat = euler2mat(-np.pi / 2.0, np.pi / 2.0, 0.0)
    right_hand_points = np.dot(hand_points, init_mat.T)
    update_pose_3 = robot_to_hand(pose_3)
    hand_ori = update_pose_3[:3, :3]
    offset = np.array([[-0.02, 0.05, -0.1]])
    offset = offset @ hand_ori.T
    hand_trans = update_pose_3[:3, 3].reshape(1, 3) + offset
    transformed_point_cloud = np.dot(right_hand_points, hand_ori.T) + hand_trans

    return transformed_point_cloud

def transform_left_leap_pointcloud_to_camera_frame(left_hand_points, left_pose_data):
    pose_3 = np.eye(4)
    pose_3[:3, :3] = quat2mat(left_pose_data[3:7])
    pose_3[:3, 3] = left_pose_data[:3]
    init_mat = euler2mat(-np.pi / 2.0, np.pi / 2.0, 0.0)
    left_hand_points = np.dot(left_hand_points, init_mat.T)
    update_pose_3 = robot_to_hand_left(pose_3)
    hand_ori = update_pose_3[:3, :3]
    offset = np.array([[0.02, 0.05, -0.1]])
    offset = offset @ hand_ori.T
    hand_trans = update_pose_3[:3, 3].reshape(1, 3) + offset
    transformed_point_cloud_left = np.dot(left_hand_points, hand_ori.T) + hand_trans

    return transformed_point_cloud_left

def mask_image(image, robot0_eef, corrected_pose, left=False):

    translation = robot0_eef[0:3]
    rotation_quat = robot0_eef[3:7]
    rotation_mat = quat2mat(rotation_quat)
    pose_3 = np.eye(4)
    pose_3[:3, :3] = rotation_mat
    pose_3[:3, 3] = translation
    corrected_pose = corrected_pose.reshape((4, 4))

    if left:
        update_pose_3 = robot_to_hand_left(pose_3)
    else:
        update_pose_3 = robot_to_hand(pose_3)
    hand_ori = update_pose_3[:3, :3]
    offset = np.array([[0.0, 0.0, -0.1]])
    offset = offset @ hand_ori.T
    hand = update_pose_3[:3, 3].reshape(1, 3) + offset

    right_hand_joint_points_homogeneous = np.hstack((hand, np.ones((hand.shape[0], 1))))
    right_hand_transformed_points_homogeneous = np.dot(right_hand_joint_points_homogeneous, np.linalg.inv(corrected_pose).T)
    right_hand_points_to_project = right_hand_transformed_points_homogeneous[:, :3] / right_hand_transformed_points_homogeneous[:, [3]]
    right_hand_back_projected_points = _back_project_batch(right_hand_points_to_project, o3d_depth_intrinsic.intrinsic_matrix)

    right_hand_back_projected_points[:, 0] = right_hand_back_projected_points[:, 0] / 1280 * 84
    right_hand_back_projected_points[:, 1] = right_hand_back_projected_points[:, 1] / 720 * 84

    cv2.circle(image, (right_hand_back_projected_points[0][0], right_hand_back_projected_points[0][1]), 20, (255, 255, 255), -1)

    if (0 <= right_hand_back_projected_points[0][0] < 84) and (0 <= right_hand_back_projected_points[0][1] < 84):
        return image, True
    else:
        return image, False

def translate_wrist_to_origin(joint_positions):
    wrist_position = joint_positions[0]
    updated_positions = joint_positions - wrist_position
    return updated_positions

def apply_pose_matrix(joint_positions, pose_matrix):
    homogeneous_joint_positions = np.hstack([joint_positions, np.ones((joint_positions.shape[0], 1))])
    transformed_positions = np.dot(homogeneous_joint_positions, pose_matrix.T)
    transformed_positions_3d = transformed_positions[:, :3]
    return transformed_positions_3d

def inverse_transformation(matrix):
    # Assuming matrix is a 4x4 numpy array
    R = matrix[:3, :3]
    T = matrix[:3, 3]

    R_inv = np.linalg.inv(R)
    T_inv = -np.dot(R_inv, T)

    inverse_matrix = np.eye(4)  # Create a 4x4 identity matrix
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = T_inv

    return inverse_matrix

def hand_to_robot(pose_data):
    global R_delta_init

    pose_ori_matirx = pose_data[:3, :3]
    pose_ori_correction_matrix = np.array([[0, -1, 0],
                                          [0, 0, 1],
                                          [1, 0, 0]])
    pose_ori_matirx = np.dot(pose_ori_matirx, pose_ori_correction_matrix)

    goal_ori = np.dot(R_delta_init, pose_ori_matirx)

    goal_pos = np.array(REALROBOT_RIGHT_HAND_OFFSET["absolute_offset"]) + \
               np.dot(np.array(REALROBOT_RIGHT_HAND_OFFSET["goal_ori_offset"]), goal_ori) + \
               np.array([-pose_data[2][3], -pose_data[0][3], pose_data[1][3]])

    return_pose_data = np.eye(4)
    return_pose_data[:3, :3] = goal_ori
    return_pose_data[:3, 3] = goal_pos

    return return_pose_data

def hand_to_robot_left(pose_data):
    global R_delta_init

    pose_ori_matirx = pose_data[:3, :3]
    pose_ori_correction_matrix = np.array([[0, -1, 0],
                                          [0, 0, 1],
                                          [1, 0, 0]])
    pose_ori_matirx = np.dot(pose_ori_matirx, pose_ori_correction_matrix)

    goal_ori = np.dot(R_delta_init, pose_ori_matirx)

    goal_pos = np.array(REALROBOT_LEFT_HAND_OFFSET["absolute_offset"]) + \
               np.dot(np.array(REALROBOT_LEFT_HAND_OFFSET["goal_ori_offset"]), goal_ori) + \
               np.array([-pose_data[2][3], -pose_data[0][3], pose_data[1][3]])

    return_pose_data = np.eye(4)
    return_pose_data[:3, :3] = goal_ori
    return_pose_data[:3, 3] = goal_pos

    return return_pose_data

def robot_to_hand(pose_data):
    global R_delta_init

    goal_pos = pose_data[:3, 3]
    goal_ori = pose_data[:3, :3]

    pose_ori_correction_matrix_inv = np.linalg.inv(np.array([[0, -1, 0],
                                                            [0, 0, 1],
                                                            [1, 0, 0]]))

    pose_ori_matrix = np.dot(np.linalg.inv(R_delta_init), goal_ori)
    pose_ori_matrix = np.dot(pose_ori_matrix, pose_ori_correction_matrix_inv)

    translation_component = np.array(REALROBOT_RIGHT_HAND_OFFSET["absolute_offset"]) + \
                            np.dot(np.array(REALROBOT_RIGHT_HAND_OFFSET["goal_ori_offset"]), goal_ori)
    pose_data_translation = goal_pos - translation_component

    return_pose_data = np.eye(4)
    return_pose_data[:3, :3] = pose_ori_matrix
    return_pose_data[:3, 3] = [-pose_data_translation[1], pose_data_translation[2], -pose_data_translation[0]]

    return return_pose_data

def robot_to_hand_left(pose_data):
    global R_delta_init

    goal_pos = pose_data[:3, 3]
    goal_ori = pose_data[:3, :3]

    pose_ori_correction_matrix_inv = np.linalg.inv(np.array([[0, -1, 0],
                                                            [0, 0, 1],
                                                            [1, 0, 0]]))

    pose_ori_matrix = np.dot(np.linalg.inv(R_delta_init), goal_ori)
    pose_ori_matrix = np.dot(pose_ori_matrix, pose_ori_correction_matrix_inv)

    translation_component = np.array(REALROBOT_LEFT_HAND_OFFSET["absolute_offset"]) + \
                            np.dot(np.array(REALROBOT_LEFT_HAND_OFFSET["goal_ori_offset"]), goal_ori)
    pose_data_translation = goal_pos - translation_component

    return_pose_data = np.eye(4)
    return_pose_data[:3, :3] = pose_ori_matrix
    return_pose_data[:3, 3] = [-pose_data_translation[1], pose_data_translation[2], -pose_data_translation[0]]

    return return_pose_data

def update_R_delta_init(frame_0_eef_pos, frame_0_eef_quat):
    global R_delta_init

    frame_0_pose = np.eye(4)
    frame_0_pose[:3, :3] = quat2mat(frame_0_eef_quat)
    frame_0_pose[:3, 3] = frame_0_eef_pos

    pose_ori_matirx = frame_0_pose[:3, :3]
    pose_ori_correction_matrix = np.dot(np.array([[0, -1, 0],
                                                  [0, 0, 1],
                                                  [1, 0, 0]]), euler2mat(0, 0, 0))
    pose_ori_matirx = np.dot(pose_ori_matirx, pose_ori_correction_matrix)

    canonical_t265_ori = np.array([[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, -1]])
    x_angle, y_angle, z_angle = mat2euler(frame_0_pose[:3, :3])
    canonical_t265_ori = np.dot(canonical_t265_ori, euler2mat(-z_angle, x_angle + 0.3, y_angle))

    R_delta_init = np.dot(canonical_t265_ori, pose_ori_matirx.T)

def rotate_quaternion(a1=0, a2=0, a3=0):
    q_transform = p.getQuaternionFromEuler([a1, a2, a3])

    return q_transform

def rotate_quaternion_xyzw(quaternion_xyzw, axis, angle):
    """
    Rotate a quaternion in the "xyzw" format along a specified axis by a given angle in radians.

    Args:
        quaternion_xyzw (np.ndarray): The input quaternion in "xyzw" format [x, y, z, w].
        axis (np.ndarray): The axis of rotation [x, y, z].
        angle (float): The rotation angle in radians.

    Returns:
        np.ndarray: The rotated quaternion in "xyzw" format [x', y', z', w'].
    """
    # Normalize the axis of rotation
    q1 = np.array(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]]
    )
    q2 = axangle2quat(axis, angle)
    q3 = qmult(q2, q1)

    rotated_quaternion_xyzw = np.array([q3[1], q3[2], q3[3], q3[0]])

    return rotated_quaternion_xyzw

def rotate_vector_by_quaternion_using_matrix(v, q):
    # Convert the quaternion to a rotation matrix
    q1 = np.array([q[3], q[0], q[1], q[2]])
    rotation_matrix = quat2mat(q1)

    homogenous_vector = np.array([v[0], v[1], v[2]]).T
    rotated_vector_homogeneous = np.dot(homogenous_vector, rotation_matrix.T)

    return rotated_vector_homogeneous

def switch_axis(quaternion_xyzw, i, j):
    q1 = np.array(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]]
    )
    rot_mat = quat2mat(q1)
    rot_mat_copy = copy.deepcopy(rot_mat)
    rot_mat[i] = rot_mat_copy[j]
    rot_mat[j] = rot_mat_copy[i]
    import pdb

    pdb.set_trace()
    q2 = mat2quat(rot_mat)
    q3 = np.array([q2[1], q2[2], q2[3], q2[0]])
    return q3

def swap_quaternion_axes(quaternion, axis1, axis2):
    """
    Swap two axes in a quaternion without converting to Euler angles.

    Args:
        quaternion (list or np.ndarray): The input quaternion [x, y, z, w].
        axis1 (int): The index of the first axis to swap (0 for X, 1 for Y, 2 for Z).
        axis2 (int): The index of the second axis to swap (0 for X, 1 for Y, 2 for Z).

    Returns:
        np.ndarray: The new quaternion with swapped axes [x', y', z', w'].
    """
    if axis1 < 0 or axis1 > 2 or axis2 < 0 or axis2 > 2:
        raise ValueError("Axis indices must be 0, 1, or 2.")

    # Create a copy of the input quaternion
    new_quaternion = quaternion.copy()

    # Swap the elements corresponding to the specified axes
    new_quaternion[axis1], new_quaternion[axis2] = quaternion[axis2], quaternion[axis1]

    return new_quaternion

def rotate_vector_by_quaternion(vector, quaternion):
    transformed_point = p.multiplyTransforms(
        [0, 0, 0], quaternion, vector, [0, 0, 0, 1]
    )[0]
    return transformed_point