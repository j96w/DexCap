import copy
import h5py
import json
import cv2
import numpy as np
import os
import transforms3d
from transforms3d.quaternions import mat2quat
from scipy.spatial.transform import Rotation
from transforms3d.axangles import axangle2mat


def resize_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (84, 84))
    return image_resized

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

def calculate_rotation_deltas(quaternions, frame_step=3):
    """
    Calculate the delta rotation in Euler angles between each quaternion and the quaternion 'frame_step' frames later.

    :param quaternions: Numpy array of quaternions [w, x, y, z]
    :param frame_step: Number of frames to look ahead for the delta calculation
    :return: Array of delta rotations in Euler angles
    """
    num_quaternions = len(quaternions)
    deltas = np.zeros((num_quaternions, 3))  # Initialize array for delta rotations

    for i in range(num_quaternions):
        if i + frame_step < num_quaternions:
            q_current = quaternions[i]
            q_next = quaternions[i + frame_step]

            # Convert quaternions to rotation matrices
            mat_current = transforms3d.quaternions.quat2mat(q_current)
            mat_next = transforms3d.quaternions.quat2mat(q_next)

            # Calculate the relative rotation matrix
            relative_mat = np.dot(np.linalg.inv(mat_current), mat_next)

            # Convert the relative rotation matrix to Euler angles
            deltas[i] = transforms3d.euler.mat2euler(relative_mat)

    return deltas

def reconstruct_quaternions(original_quaternions, delta_euler_rotations):
    """
    Reconstruct a list of transformed quaternions using the original quaternion list and delta Euler rotations.

    :param original_quaternions: Numpy array of original quaternions [w, x, y, z]
    :param delta_euler_rotations: Numpy array of delta rotations in Euler angles
    :return: Numpy array of transformed quaternions
    """
    transformed_quaternions = np.zeros_like(original_quaternions)

    for i in range(len(original_quaternions)):
        # Convert delta Euler rotation to quaternion
        delta_quat = transforms3d.euler.euler2quat(*delta_euler_rotations[i])

        # Multiply original quaternion with delta quaternion
        transformed_quat = transforms3d.quaternions.qmult(original_quaternions[i], delta_quat)
        transformed_quaternions[i] = transformed_quat

    return transformed_quaternions

def calculate_translation_deltas(translations, frame_step=3):
    """
    Calculate the delta translation between each position and the position 'frame_step' frames later.

    :param translations: Numpy array of translations
    :param frame_step: Number of frames to look ahead for the delta calculation
    :return: Array of delta translations
    """
    num_translations = len(translations)
    deltas = np.zeros_like(translations)  # Initialize array for delta translations

    for i in range(num_translations):
        if i + frame_step < num_translations:
            deltas[i] = translations[i + frame_step] - translations[i]

    return deltas


def _back_project_point(point, intrinsics):
    """ Back-project a single point from 3D to 2D image space """
    x, y, z = point
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    return int(u), int(v)