import os
import time
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
import cv2
from pynput import keyboard
from scipy.spatial.transform import Rotation

# Default hyperparameters
DEFAULT_CAM_TO_VIVE = np.array([
    [-1,  0,  0,  0.04],
    [ 0,  0,  1,  0.05],
    [ 0,  1,  0, -0.08],
    [ 0,  0,  0,  1]
])

DEFAULT_LEFT_GLOVE_TO_VIVE = np.array([
    [ 1,  0,  0,  0.0],
    [ 0,  0, -1,  -0.01],
    [ 0, -1,  0,  -0.05],
    [ 0,  0,  0,  1]
])

DEFAULT_RIGHT_GLOVE_TO_VIVE = np.array([
    [ 1,  0,  0,  0.02],
    [ 0,  0, -1, -0.01],
    [ 0, -1,  0, -0.07],
    [ 0,  0,  0,  1]
])

# Global variables
CAM_TO_VIVE = DEFAULT_CAM_TO_VIVE.copy()
LEFT_GLOVE_TO_VIVE = DEFAULT_LEFT_GLOVE_TO_VIVE.copy()
RIGHT_GLOVE_TO_VIVE = DEFAULT_RIGHT_GLOVE_TO_VIVE.copy()
adjusting_glove = 0  # 0: CAM_TO_VIVE, 1: LEFT_GLOVE_TO_VIVE, 2: RIGHT_GLOVE_TO_VIVE
use_fixed_frame = False
fixed_frame_id = 0

def load_pose(file_path):
    with open(file_path, 'r') as f:
        data = f.read().strip().split()
        translation = list(map(float, data[:3]))
        rotation = list(map(float, data[3:]))
    return translation, rotation

def load_intrinsics(file_path):
    intrinsics = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split()
            intrinsics[key] = float(value)
    return intrinsics

def load_transformation(file_path, default_transformation):
    if os.path.exists(file_path):
        return np.loadtxt(file_path)
    else:
        return default_transformation

def save_transformation(file_path, transformation):
    np.savetxt(file_path, transformation)

def create_point_cloud_from_images(color_image, depth_image, intrinsics):
    height, width = depth_image.shape

    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy']
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    depth_image_o3d = o3d.geometry.Image(depth_image)
    color_image_o3d = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image_o3d, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    return pcd

def create_cube(color):
    cube = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
    cube.paint_uniform_color(color)
    return cube

def set_cube_pose(cube, translation, quaternion):
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(quaternion).rotation_matrix
    transform[:3, 3] = translation
    cube.transform(transform)

def reset_cube_pose(cube, initial_transform):
    cube.transform(np.linalg.inv(initial_transform))

def get_transformation_matrix(translation, quaternion):
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(quaternion).rotation_matrix
    transform[:3, 3] = translation
    return transform

def translate_wrist_to_origin(joint_positions):
    wrist_position = joint_positions[0]
    updated_positions = joint_positions - wrist_position
    return updated_positions

def process_hand_joints(joint_xyz, joint_ori, glove_to_vive):
    joint_xyz = translate_wrist_to_origin(joint_xyz)
    wrist_ori = joint_ori[0]
    rotation_matrix = Rotation.from_quat(wrist_ori).as_matrix().T
    joint_xyz_reshaped = joint_xyz[:, :, np.newaxis]
    transformed_joint_xyz = np.matmul(rotation_matrix, joint_xyz_reshaped)
    joint_xyz = transformed_joint_xyz[:, :, 0]

    # From glove frame to VIVE frame
    joint_xyz = np.dot(joint_xyz, glove_to_vive[:3, :3].T)
    joint_xyz += np.array([glove_to_vive[0, 3], glove_to_vive[1, 3], glove_to_vive[2, 3]])

    return joint_xyz

def on_press(key):
    global CAM_TO_VIVE, LEFT_GLOVE_TO_VIVE, RIGHT_GLOVE_TO_VIVE, adjusting_glove, use_fixed_frame
    try:
        if key.char == 'm':
            adjusting_glove = (adjusting_glove + 1) % 3
            if adjusting_glove == 0:
                print("Adjusting CAM_TO_VIVE")
            elif adjusting_glove == 1:
                print("Adjusting LEFT_GLOVE_TO_VIVE")
            elif adjusting_glove == 2:
                print("Adjusting RIGHT_GLOVE_TO_VIVE")
        elif key.char == 'n':
            use_fixed_frame = not use_fixed_frame
            print(f"Using fixed frame: {use_fixed_frame}")
        elif adjusting_glove == 1:
            if key.char == 'i':
                LEFT_GLOVE_TO_VIVE[0, 3] += 0.01
            elif key.char == 'k':
                LEFT_GLOVE_TO_VIVE[0, 3] -= 0.01
            elif key.char == 'j':
                LEFT_GLOVE_TO_VIVE[1, 3] -= 0.01
            elif key.char == 'l':
                LEFT_GLOVE_TO_VIVE[1, 3] += 0.01
            elif key.char == 'u':
                LEFT_GLOVE_TO_VIVE[2, 3] += 0.01
            elif key.char == 'o':
                LEFT_GLOVE_TO_VIVE[2, 3] -= 0.01
            print(f"LEFT_GLOVE_TO_VIVE: {LEFT_GLOVE_TO_VIVE}")
        elif adjusting_glove == 2:
            if key.char == 'i':
                RIGHT_GLOVE_TO_VIVE[0, 3] += 0.01
            elif key.char == 'k':
                RIGHT_GLOVE_TO_VIVE[0, 3] -= 0.01
            elif key.char == 'j':
                RIGHT_GLOVE_TO_VIVE[1, 3] -= 0.01
            elif key.char == 'l':
                RIGHT_GLOVE_TO_VIVE[1, 3] += 0.01
            elif key.char == 'u':
                RIGHT_GLOVE_TO_VIVE[2, 3] += 0.01
            elif key.char == 'o':
                RIGHT_GLOVE_TO_VIVE[2, 3] -= 0.01
            print(f"RIGHT_GLOVE_TO_VIVE: {RIGHT_GLOVE_TO_VIVE}")
        else:
            if key.char == 'i':
                CAM_TO_VIVE[0, 3] += 0.01
            elif key.char == 'k':
                CAM_TO_VIVE[0, 3] -= 0.01
            elif key.char == 'j':
                CAM_TO_VIVE[1, 3] -= 0.01
            elif key.char == 'l':
                CAM_TO_VIVE[1, 3] += 0.01
            elif key.char == 'u':
                CAM_TO_VIVE[2, 3] += 0.01
            elif key.char == 'o':
                CAM_TO_VIVE[2, 3] -= 0.01
            print(f"CAM_TO_VIVE: {CAM_TO_VIVE}")
    except AttributeError:
        pass

def create_sphere(radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    return sphere

def visualize_frames(dataset_folder, intrinsics):
    global cam_to_vive_file_path, left_glove_to_vive_file_path, right_glove_to_vive_file_path, CAM_TO_VIVE, LEFT_GLOVE_TO_VIVE, RIGHT_GLOVE_TO_VIVE, use_fixed_frame, fixed_frame_id
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    data_folder = os.path.join(dataset_folder, "data")
    frame_folders = sorted([os.path.join(data_folder, d) for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))])

    pcd = None
    right_cube = create_cube([1, 0, 0])
    left_cube = create_cube([0, 0, 1])
    chest_cube = create_cube([0, 1, 0])

    vis.add_geometry(right_cube)
    vis.add_geometry(left_cube)
    vis.add_geometry(chest_cube)

    left_hand_points = [create_sphere(0.01, [0, 0, 1]) for _ in range(21)]
    right_hand_points = [create_sphere(0.01, [1, 0, 0]) for _ in range(21)]

    for sphere in left_hand_points + right_hand_points:
        vis.add_geometry(sphere)

    initial_transform_right = np.eye(4)
    initial_transform_left = np.eye(4)
    initial_transform_chest = np.eye(4)

    cam_to_vive_file_path = os.path.join(dataset_folder, 'cam_to_vive.txt')
    left_glove_to_vive_file_path = os.path.join(dataset_folder, 'left_glove_to_vive.txt')
    right_glove_to_vive_file_path = os.path.join(dataset_folder, 'right_glove_to_vive.txt')

    CAM_TO_VIVE = load_transformation(cam_to_vive_file_path, DEFAULT_CAM_TO_VIVE)
    LEFT_GLOVE_TO_VIVE = load_transformation(left_glove_to_vive_file_path, DEFAULT_LEFT_GLOVE_TO_VIVE)
    RIGHT_GLOVE_TO_VIVE = load_transformation(right_glove_to_vive_file_path, DEFAULT_RIGHT_GLOVE_TO_VIVE)

    while True:
        frame_i = fixed_frame_id if use_fixed_frame else (fixed_frame_id % len(frame_folders))
        frame_folder = frame_folders[frame_i]
        frame_pcd_folder = frame_folders[frame_i]

        color_image_path = os.path.join(frame_pcd_folder, "color.png")
        depth_image_path = os.path.join(frame_pcd_folder, "depth.png")

        color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        right_pose_path = os.path.join(frame_folder, "right_pose.txt")
        left_pose_path = os.path.join(frame_folder, "left_pose.txt")
        chest_pose_path = os.path.join(frame_folder, "chest_pose.txt")

        left_hand_pos_path = os.path.join(frame_folder, "raw_left_hand_joint_xyz.txt")
        left_hand_ori_path = os.path.join(frame_folder, "raw_left_hand_joint_orientation.txt")
        right_hand_pos_path = os.path.join(frame_folder, "raw_right_hand_joint_xyz.txt")
        right_hand_ori_path = os.path.join(frame_folder, "raw_right_hand_joint_orientation.txt")

        if os.path.exists(left_hand_pos_path) and os.path.exists(left_hand_ori_path):
            left_hand_joint_xyz = np.loadtxt(left_hand_pos_path)
            left_hand_joint_ori = np.loadtxt(left_hand_ori_path)
            left_hand_joint_xyz = process_hand_joints(left_hand_joint_xyz, left_hand_joint_ori, LEFT_GLOVE_TO_VIVE)

        if os.path.exists(right_hand_pos_path) and os.path.exists(right_hand_ori_path):
            right_hand_joint_xyz = np.loadtxt(right_hand_pos_path)
            right_hand_joint_ori = np.loadtxt(right_hand_ori_path)
            right_hand_joint_xyz = process_hand_joints(right_hand_joint_xyz, right_hand_joint_ori, RIGHT_GLOVE_TO_VIVE)

        if os.path.exists(chest_pose_path):
            chest_translation, chest_rotation = load_pose(chest_pose_path)
            chest_transform = get_transformation_matrix(chest_translation, chest_rotation)

            new_pcd = create_point_cloud_from_images(color_image, depth_image, intrinsics)

            # Apply transformations
            new_pcd.transform(CAM_TO_VIVE)  # From cam frame to VIVE frame
            new_pcd.transform(chest_transform)  # From VIVE frame to world frame

            save_transformation(cam_to_vive_file_path, CAM_TO_VIVE)
            save_transformation(left_glove_to_vive_file_path, LEFT_GLOVE_TO_VIVE)
            save_transformation(right_glove_to_vive_file_path, RIGHT_GLOVE_TO_VIVE)

            if pcd is None:
                pcd = new_pcd
                vis.add_geometry(pcd)
            else:
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors

            reset_cube_pose(chest_cube, initial_transform_chest)
            set_cube_pose(chest_cube, chest_translation, chest_rotation)
            initial_transform_chest = chest_transform

        if os.path.exists(right_pose_path):
            right_translation, right_rotation = load_pose(right_pose_path)
            reset_cube_pose(right_cube, initial_transform_right)
            set_cube_pose(right_cube, right_translation, right_rotation)
            initial_transform_right[:3, :3] = Quaternion(right_rotation).rotation_matrix
            initial_transform_right[:3, 3] = right_translation

            if os.path.exists(right_hand_pos_path):
                right_hand_transform = get_transformation_matrix(right_translation, right_rotation)
                for i, (sphere, pos) in enumerate(zip(right_hand_points, right_hand_joint_xyz)):
                    sphere.translate(-sphere.get_center())  # Reset to origin
                    sphere.translate(pos)
                    sphere.transform(right_hand_transform)

        if os.path.exists(left_pose_path):
            left_translation, left_rotation = load_pose(left_pose_path)
            reset_cube_pose(left_cube, initial_transform_left)
            set_cube_pose(left_cube, left_translation, left_rotation)
            initial_transform_left[:3, :3] = Quaternion(left_rotation).rotation_matrix
            initial_transform_left[:3, 3] = left_translation

            if os.path.exists(left_hand_pos_path):
                left_hand_transform = get_transformation_matrix(left_translation, left_rotation)
                for i, (sphere, pos) in enumerate(zip(left_hand_points, left_hand_joint_xyz)):
                    sphere.translate(-sphere.get_center())  # Reset to origin
                    sphere.translate(pos)
                    sphere.transform(left_hand_transform)

        vis.update_geometry(pcd)
        vis.update_geometry(right_cube)
        vis.update_geometry(left_cube)
        vis.update_geometry(chest_cube)

        for sphere in left_hand_points + right_hand_points:
            vis.update_geometry(sphere)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.1)  # Adjust the delay as needed to control the frame rate

        if not use_fixed_frame:
            fixed_frame_id += 1

    vis.run()
    vis.destroy_window()
    listener.stop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Frames from Dataset")
    parser.add_argument("dataset_folder", type=str, help="Folder containing the dataset")
    args = parser.parse_args()

    intrinsics_file = os.path.join(args.dataset_folder, "camera_matrix.txt")
    intrinsics = load_intrinsics(intrinsics_file)

    visualize_frames(args.dataset_folder, intrinsics)
