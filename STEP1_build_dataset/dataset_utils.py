import os.path

import h5py
import json
from scipy.linalg import svd
from data_utils import *
#from hyperparameters import *
import copy
import cv2 as cv
from datetime import datetime
R_delta_init = None

def read_pose_data(frame_path, demo_path, fixed_trans_to_robot_table, first_frame=False):
    global leapPybulletIK

    cam_pose_path = os.path.join(frame_path, "pose.txt")

    # load left hand pose
    left_pose_path = os.path.join(frame_path, "pose_2.txt")
    left_hand_pos_path = os.path.join(frame_path, "left_hand_joint.txt")
    left_hand_ori_path = os.path.join(frame_path, "left_hand_joint_ori.txt")
    left_hand_off_path = os.path.join(demo_path, "calib_offset_left.txt")
    left_hand_off_ori_path = os.path.join(demo_path, "calib_ori_offset_left.txt")

    pose_2 = np.loadtxt(left_pose_path)
    pose_2[:3, 3] += fixed_trans_to_robot_table.T
    pose_2 = pose_2 @ between_cam_2

    left_hand_joint_xyz = np.loadtxt(left_hand_pos_path)
    left_hand_joint_xyz = translate_wrist_to_origin(left_hand_joint_xyz)  # canonical view by translate to origin
    left_hand_wrist_ori = np.loadtxt(left_hand_ori_path)[0]

    # load right hand pose
    pose_path = os.path.join(frame_path, "pose_3.txt")
    hand_pos_path = os.path.join(frame_path, "right_hand_joint.txt")
    hand_ori_path = os.path.join(frame_path, "right_hand_joint_ori.txt")
    hand_off_path = os.path.join(demo_path, "calib_offset.txt")
    hand_off_ori_path = os.path.join(demo_path, "calib_ori_offset.txt")

    pose_3 = np.loadtxt(pose_path)
    pose_3[:3, 3] += fixed_trans_to_robot_table.T
    pose_3 = pose_3 @ between_cam_3

    right_hand_joint_xyz = np.loadtxt(hand_pos_path)
    right_hand_joint_xyz = translate_wrist_to_origin(right_hand_joint_xyz)  # canonical view by translate to origin
    right_hand_wrist_ori = np.loadtxt(hand_ori_path)[0]

    right_hand_target, left_hand_target, right_hand_points, left_hand_points = leapPybulletIK.compute_IK(right_hand_joint_xyz, right_hand_wrist_ori, left_hand_joint_xyz, left_hand_wrist_ori)
    np.savetxt(os.path.join(frame_path, "right_joints.txt"), right_hand_target)
    np.savetxt(os.path.join(frame_path, "left_joints.txt"), left_hand_target)

    # convert left hand pose
    left_rotation_matrix = Rotation.from_quat(left_hand_wrist_ori).as_matrix().T
    left_joint_xyz_reshaped = left_hand_joint_xyz[:, :, np.newaxis]
    left_transformed_joint_xyz = np.matmul(left_rotation_matrix, left_joint_xyz_reshaped)
    left_hand_joint_xyz = left_transformed_joint_xyz[:, :, 0]
    left_hand_joint_xyz[:, -1] = -left_hand_joint_xyz[:, -1]  # z-axis revert
    rotation_matrix = axangle2mat(np.array([0, 1, 0]), -np.pi * 1 / 2)  # y-axis rotate
    left_hand_joint_xyz = np.dot(left_hand_joint_xyz, rotation_matrix.T)
    rotation_matrix = axangle2mat(np.array([1, 0, 0]), np.pi * 1 / 2)  # x-axis rotate
    left_hand_joint_xyz = np.dot(left_hand_joint_xyz, rotation_matrix.T)
    rotation_matrix = axangle2mat(np.array([0, 0, 1]), -np.pi * 1 / 2)  # z-axis rotate
    left_hand_joint_xyz = np.dot(left_hand_joint_xyz, rotation_matrix.T)
    left_hand_ori_offset = np.loadtxt(left_hand_off_ori_path)
    left_hand_joint_xyz = np.dot(left_hand_joint_xyz, euler2mat(*left_hand_ori_offset).T)  # rotation calibration
    left_hand_offset = np.loadtxt(left_hand_off_path)
    left_hand_joint_xyz += left_hand_offset
    left_hand_joint_xyz = apply_pose_matrix(left_hand_joint_xyz, pose_2)

    update_pose_2 = copy.deepcopy(pose_2)
    update_pose_2[:3, 3] = left_hand_joint_xyz[0]

    left_hand_joint_xyz = apply_pose_matrix(left_hand_joint_xyz, inverse_transformation(update_pose_2))

    # important! since the camera mount on the gloves are 45 degree facing up, we need to convert that 45 degree here to get the correct hand orientation
    rotation_45lookup_matrix = axangle2mat(np.array([1, 0, 0]), np.pi * 1 / 4)  # z-axis rotate
    update_pose_2[:3, :3] = np.dot(update_pose_2[:3, :3], rotation_45lookup_matrix.T)

    if not first_frame:
        update_pose_2 = hand_to_robot_left(update_pose_2)

    left_hand_translation = update_pose_2[:3, 3]
    left_hand_rotation_matrix = update_pose_2[:3, :3]

    left_hand_quaternion = mat2quat(left_hand_rotation_matrix)

    # convert right hand pose
    right_rotation_matrix = Rotation.from_quat(right_hand_wrist_ori).as_matrix().T
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
    right_hand_ori_offset = np.loadtxt(hand_off_ori_path)
    right_hand_joint_xyz = np.dot(right_hand_joint_xyz, euler2mat(*right_hand_ori_offset).T)  # rotation calibration
    right_hand_offset = np.loadtxt(hand_off_path)
    right_hand_joint_xyz += right_hand_offset
    right_hand_joint_xyz = apply_pose_matrix(right_hand_joint_xyz, pose_3)

    update_pose_3 = copy.deepcopy(pose_3)
    update_pose_3[:3, 3] = right_hand_joint_xyz[0]

    right_hand_joint_xyz = apply_pose_matrix(right_hand_joint_xyz, inverse_transformation(update_pose_3))

    # important! since the camera mount on the gloves are 45 degree facing up, we need to convert that 45 degree here to get the correct hand orientation
    rotation_45lookup_matrix = axangle2mat(np.array([1, 0, 0]), np.pi * 1 / 4)  # z-axis rotate
    update_pose_3[:3, :3] = np.dot(update_pose_3[:3, :3], rotation_45lookup_matrix.T)

    if not first_frame:
        update_pose_3 = hand_to_robot(update_pose_3)

    right_hand_translation = update_pose_3[:3, 3]
    right_hand_rotation_matrix = update_pose_3[:3, :3]

    right_hand_quaternion = mat2quat(right_hand_rotation_matrix)

    cam_pose_4x4 = np.loadtxt(cam_pose_path)
    cam_pose_4x4[:3, 3] += fixed_trans_to_robot_table.T

    cam_corrected_pose = cam_pose_4x4 @ between_cam
    cam_corrected_pose = cam_corrected_pose.flatten()

    return (np.concatenate([right_hand_translation, right_hand_quaternion, right_hand_target]),
            np.concatenate([left_hand_translation, left_hand_quaternion, left_hand_target]),
            cam_corrected_pose,
            right_hand_joint_xyz.flatten(),
            left_hand_joint_xyz.flatten(),
            right_hand_points,
            left_hand_points)

def crop_pcd(pcd, min_bound, max_bound):
    pcd_pos = pcd[:,:3]
    pcd_color = pcd[:,3:]
    mask = np.logical_and(np.all(pcd_pos > min_bound, axis=1), np.all(pcd_pos < max_bound, axis=1))
    pcd_pos = pcd_pos[mask]
    pcd_color = pcd_color[mask]
    return np.hstack([pcd_pos, pcd_color])

def process_hdf5(output_hdf5_file, dataset_folders, action_gap, num_points_to_sample, in_wild_data=False):
    global R_delta_init

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_vis = o3d.geometry.PointCloud()  # Empty point cloud for starters
    firstfirst = True

    with h5py.File(output_hdf5_file, 'w') as output_hdf5:
        output_data_group = output_hdf5.create_group('data')

        demo_index = 0
        total_frames = 0
        mean_init_pos = []
        mean_init_quat = []
        mean_init_hand = []

        for dataset_folder in dataset_folders:
            clip_marks_json = os.path.join(dataset_folder, 'clip_marks.json')

            if in_wild_data: # if the data is collected in the wild, read the fix translation to the robot table
                fixed_trans_to_robot_table = np.loadtxt(os.path.join(dataset_folder, 'map_to_robot_table_trans.txt'))
            else:
                fixed_trans_to_robot_table = np.array([0.0, 0.0, 0.0])

            # Load clip marks
            with open(clip_marks_json, 'r') as file:
                clip_marks = json.load(file)

            for clip in clip_marks:

                # save frame0 & update R_delta_init
                frame0_pose_data, frame0_left_pose_data, _, _, _, _, _ = read_pose_data(os.path.join(dataset_folder, f'frame_0'), dataset_folder, fixed_trans_to_robot_table=fixed_trans_to_robot_table, first_frame=True)
                update_R_delta_init(frame0_pose_data[:3], frame0_pose_data[3:7])

                # Get start and end frame numbers
                start_frame = int(clip['start'].split('_')[-1])
                end_frame = int(clip['end'].split('_')[-1])
                clip_length = end_frame - start_frame + 1 # include frame 0

                agentview_images = []
                pointcloud = []
                poses = []
                poses_left = []
                states = []
                glove_states = []
                left_glove_states = []
                labels = []

                for frame_number in list(range(start_frame, end_frame + 1)):

                    frame_folder = f'frame_{frame_number}'
                    image_path = os.path.join(dataset_folder, frame_folder, "color_image.jpg")
                    frame_path = os.path.join(dataset_folder, frame_folder)

                    # load hand pose data
                    pose_data, left_pose_data, cam_data, glove_data, left_glove_data, right_hand_points, left_hand_points = read_pose_data(frame_path, dataset_folder, fixed_trans_to_robot_table=fixed_trans_to_robot_table)
                    poses.append(pose_data)
                    poses_left.append(left_pose_data)

                    states.append(cam_data)
                    glove_states.append(glove_data)
                    left_glove_states.append(left_glove_data)

                    # process image
                    resized_image = resize_image(image_path)
                    resized_image, right_hand_show = mask_image(resized_image, pose_data, cam_data)
                    resized_image, left_hand_show = mask_image(resized_image, left_pose_data, cam_data, left=True)
                    agentview_images.append(resized_image)

                    # process pointcloud
                    color_image_o3d = o3d.io.read_image(os.path.join(dataset_folder, frame_folder, "color_image.jpg"))
                    depth_image_o3d = o3d.io.read_image(os.path.join(dataset_folder, frame_folder, "depth_image.png"))
                    max_depth = 1000
                    depth_array = np.asarray(depth_image_o3d)
                    mask = depth_array > max_depth
                    depth_array[mask] = 0
                    filtered_depth_image = o3d.geometry.Image(depth_array)
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, filtered_depth_image, depth_trunc=4.0, convert_rgb_to_intensity=False)

                    pose_4x4 = np.loadtxt(os.path.join(dataset_folder, frame_folder, "pose.txt"))
                    pose_4x4[:3, 3] += fixed_trans_to_robot_table.T

                    corrected_pose = pose_4x4 @ between_cam
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_depth_intrinsic)
                    pcd.transform(corrected_pose)
                    color_pcd = np.concatenate((np.array(pcd.points), np.array(pcd.colors)), axis=-1)

                    if right_hand_show: # detected right hand in the view, merge right leap hand pointcloud into inputs
                        transformed_point_cloud = transform_right_leap_pointcloud_to_camera_frame(right_hand_points, pose_data)

                        colored_hand_point_cloud = np.concatenate((transformed_point_cloud, np.zeros((transformed_point_cloud.shape[0], 3))), axis=1)
                        color_pcd = np.concatenate((color_pcd, colored_hand_point_cloud), axis=0)

                    if left_hand_show: # detected left hand in the view, merge left leap hand pointcloud into inputs
                        transformed_point_cloud_left = transform_left_leap_pointcloud_to_camera_frame(left_hand_points, left_pose_data)

                        colored_hand_point_cloud_left = np.concatenate((transformed_point_cloud_left, np.zeros((transformed_point_cloud_left.shape[0], 3))), axis=1)
                        color_pcd = np.concatenate((color_pcd, colored_hand_point_cloud_left), axis=0)

                    # remove the redundant points bellow the table surface and background
                    centroid = np.mean(robot_table_corner_points, axis=0)
                    A = robot_table_corner_points - centroid
                    U, S, Vt = svd(A)
                    normal = Vt[-1]
                    d = -np.dot(normal, centroid)
                    xyz = color_pcd[:, :3]
                    for plane_gap in table_sweep_list: # sweep over the plane height
                        below_plane = np.dot(xyz, normal[:3]) + d + plane_gap < 0
                        if len(color_pcd[~below_plane]) > num_points_to_sample:
                            color_pcd = color_pcd[~below_plane]
                            break

                    # down sample input pointcloud
                    if len(color_pcd) > num_points_to_sample:
                        indices = np.random.choice(len(color_pcd), num_points_to_sample, replace=False)
                        color_pcd = color_pcd[indices]

                    pointcloud.append(copy.deepcopy(color_pcd))
                    labels.append(0)

                    # update pointcloud visualization
                    pcd_vis.points = o3d.utility.Vector3dVector(color_pcd[:, :3])
                    pcd_vis.colors = o3d.utility.Vector3dVector(color_pcd[:, 3:])

                    if firstfirst:
                        vis.add_geometry(pcd_vis)
                        firstfirst = False
                    else:
                        vis.update_geometry(pcd_vis)
                    vis.poll_events()
                    vis.update_renderer()

                    # update image visualization
                    cv2.imshow("masked_resized_image", resized_image)
                    cv2.waitKey(1)

                poses = np.array(poses)
                robot0_eef_pos = poses[:, :3]
                robot0_eef_quat = poses[:, 3:7]
                robot0_eef_hand = (poses[:, 7:] - np.pi) * 0.5 # scale the hand joint positions

                poses_left = np.array(poses_left)
                robot0_eef_pos_left = poses_left[:, :3]
                robot0_eef_quat_left = poses_left[:, 3:7]
                robot0_eef_hand_left = (poses_left[:, 7:] - np.pi) * 0.5 # scale the hand joint positions

                robot0_eef_pos = np.concatenate((robot0_eef_pos, robot0_eef_pos_left), axis=-1)
                robot0_eef_quat = np.concatenate((robot0_eef_quat, robot0_eef_quat_left), axis=-1)
                robot0_eef_hand = np.concatenate((robot0_eef_hand, robot0_eef_hand_left), axis=-1)

                actions_pos = np.concatenate((robot0_eef_pos[action_gap:], robot0_eef_pos[-1:].repeat(action_gap, axis=0)), axis=0)
                actions_rot = np.concatenate((robot0_eef_quat[action_gap:], robot0_eef_quat[-1:].repeat(action_gap, axis=0)), axis=0)
                actions_hand = np.concatenate((robot0_eef_hand[action_gap:], robot0_eef_hand[-1:].repeat(action_gap, axis=0)), axis=0)

                actions = np.concatenate((actions_pos, actions_rot, actions_hand), axis=-1) # merge arm and hand actions

                for j in range(action_gap): # Based on the action_gap, generate the trajectories
                    demo_name = f'demo_{demo_index}'
                    output_demo_group = output_data_group.create_group(demo_name)
                    print("{} saved".format(demo_name))
                    demo_index += 1

                    output_demo_group.attrs['frame_0_eef_pos'] = frame0_pose_data[:3]
                    output_demo_group.attrs['frame_0_eef_quat'] = frame0_pose_data[3:7]

                    output_obs_group = output_demo_group.create_group('obs')
                    output_obs_group.create_dataset('agentview_image', data=np.array(agentview_images)[j::action_gap])
                    output_obs_group.create_dataset('pointcloud', data=np.array(pointcloud)[j::action_gap])
                    output_obs_group.create_dataset('robot0_eef_pos', data=copy.deepcopy(robot0_eef_pos)[j::action_gap])
                    output_obs_group.create_dataset('robot0_eef_quat', data=copy.deepcopy(robot0_eef_quat)[j::action_gap])
                    output_obs_group.create_dataset('robot0_eef_hand', data=copy.deepcopy(robot0_eef_hand)[j::action_gap])

                    output_obs_group.create_dataset('label', data=np.array(labels)[j::action_gap])
                    output_demo_group.create_dataset('actions', data=copy.deepcopy(actions)[j::action_gap])

                    # Create 'dones', 'rewards', and 'states'
                    dones = np.zeros(clip_length, dtype=np.int64)
                    dones[-1] = 1  # Set last frame's 'done' to 1
                    output_demo_group.create_dataset('dones', data=dones[j::action_gap])

                    rewards = np.zeros(clip_length, dtype=np.float64)
                    output_demo_group.create_dataset('rewards', data=rewards[j::action_gap])
                    output_demo_group.create_dataset('states', data=states[j::action_gap])
                    output_demo_group.create_dataset('glove_states', data=glove_states[j::action_gap])

                    output_demo_group.attrs['num_samples'] = len(actions[j::action_gap])

                    total_frames += len(actions[j::action_gap])

                    mean_init_pos.append(copy.deepcopy(robot0_eef_pos[j]))
                    mean_init_quat.append(copy.deepcopy(robot0_eef_quat[j]))
                    mean_init_hand.append(copy.deepcopy(robot0_eef_hand[j]))

        output_data_group.attrs['total'] = total_frames

        # calculate the mean of the initial starting position
        mean_init_pos = np.array(mean_init_pos).mean(axis=0)
        mean_init_quat = mean_init_quat[0]
        mean_init_hand = np.array(mean_init_hand).mean(axis=0)
        output_data_group.attrs['mean_init_pos'] = mean_init_pos
        output_data_group.attrs['mean_init_quat'] = mean_init_quat
        output_data_group.attrs['mean_init_hand'] = mean_init_hand


def process_hdf5_arcap(output_hdf5_file, dataset_folder, action_gap, num_points_to_sample):
    global R_delta_init

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_vis = o3d.geometry.PointCloud()  # Empty point cloud for starters
    firstfirst = True

    with h5py.File(output_hdf5_file, 'w') as output_hdf5:
        output_data_group = output_hdf5.create_group('data')

        demo_index = 0
        total_frames = 0
        mean_init_arm = []
        mean_init_hand = []

        demos = os.listdir(dataset_folder)
        demos = sorted(demos, key=lambda x: int(x.split('_')[1]))

        for demo in demos:

            demo_path = os.path.join(dataset_folder, demo)

            pointcloud = []
            arm_joints = []
            hand_joints = []

            frames = os.listdir(demo_path)
            frames = sorted(frames, key=lambda x: int(x.split('_')[1]))

            clip_length = len(frames)

            for frame in frames:

                pointcloud_path = os.path.join(dataset_folder, demo, frame, "point_cloud.ply")
                arm_joints_path = os.path.join(dataset_folder, demo, frame, "arm_joints.txt")
                hand_joints_path = os.path.join(dataset_folder, demo, frame, "hand_joints.txt")

                ply = o3d.io.read_point_cloud(pointcloud_path)
                points = np.asarray(ply.points)
                colors = np.asarray(ply.colors)
                point_cloud_data = np.hstack((points, colors))

                # Check the number of points in the dataset
                if point_cloud_data.shape[0] < num_points_to_sample:
                    # Calculate how many times to repeat the dataset
                    repeat_count = num_points_to_sample // point_cloud_data.shape[0] + 1
                    extended_point_cloud_data = np.tile(point_cloud_data, (repeat_count, 1))
                    # Now trim the extended dataset to exactly 10000 points if it exceeds
                    point_cloud_data = extended_point_cloud_data[:num_points_to_sample]
                else:
                    # Randomly select 10000 points if there are enough
                    indices = np.random.choice(point_cloud_data.shape[0], num_points_to_sample, replace=False)
                    point_cloud_data = point_cloud_data[indices]

                arm_joints_data = np.loadtxt(arm_joints_path)
                hand_joints_data = np.loadtxt(hand_joints_path)

                pointcloud.append(copy.deepcopy(point_cloud_data))
                arm_joints.append(copy.deepcopy(arm_joints_data))
                hand_joints.append(copy.deepcopy(hand_joints_data))

                # update pointcloud visualization
                pcd_vis.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
                pcd_vis.colors = o3d.utility.Vector3dVector(point_cloud_data[:, 3:])

                if firstfirst:
                    vis.add_geometry(pcd_vis)
                    firstfirst = False
                else:
                    vis.update_geometry(pcd_vis)
                vis.poll_events()
                vis.update_renderer()

            pointcloud = np.array(pointcloud)
            arm_joints = np.array(arm_joints)
            hand_joints = np.array(hand_joints)

            actions_arm = np.concatenate((arm_joints[action_gap:], arm_joints[-1:].repeat(action_gap, axis=0)), axis=0)
            actions_hand = np.concatenate((hand_joints[action_gap:], hand_joints[-1:].repeat(action_gap, axis=0)), axis=0)

            actions = np.concatenate((actions_arm, actions_hand), axis=-1) # merge arm and hand actions

            for j in range(action_gap): # Based on the action_gap, generate the trajectories
                demo_name = 'demo_{}'.format(demo_index)
                output_demo_group = output_data_group.create_group(demo_name)
                print("{} saved".format(demo_name))
                demo_index += 1

                output_obs_group = output_demo_group.create_group('obs')
                output_obs_group.create_dataset('pointcloud', data=np.array(pointcloud)[j::action_gap])
                output_obs_group.create_dataset('robot0_arm_joints', data=copy.deepcopy(arm_joints)[j::action_gap])
                output_obs_group.create_dataset('robot0_hand_joints', data=copy.deepcopy(hand_joints)[j::action_gap])

                output_demo_group.create_dataset('actions', data=copy.deepcopy(actions)[j::action_gap])

                # Create 'dones', 'rewards', and 'states'
                dones = np.zeros(clip_length, dtype=np.int64)
                dones[-1] = 1  # Set last frame's 'done' to 1
                output_demo_group.create_dataset('dones', data=dones[j::action_gap])

                rewards = np.zeros(clip_length, dtype=np.float64)
                output_demo_group.create_dataset('rewards', data=rewards[j::action_gap])
                states = np.zeros(clip_length, dtype=np.float64)
                output_demo_group.create_dataset('states', data=states[j::action_gap])

                output_demo_group.attrs['num_samples'] = len(actions[j::action_gap])

                total_frames += len(actions[j::action_gap])

                mean_init_arm.append(copy.deepcopy(arm_joints[j]))
                mean_init_hand.append(copy.deepcopy(hand_joints[j]))

        output_data_group.attrs['total'] = total_frames

        # calculate the mean of the initial starting position
        mean_init_arm = np.array(mean_init_arm).mean(axis=0)
        mean_init_hand = np.array(mean_init_hand).mean(axis=0)
        output_data_group.attrs['mean_init_arm'] = mean_init_arm
        output_data_group.attrs['mean_init_hand'] = mean_init_hand

# Added wrist pose
def process_hdf5_arcap_multi(output_hdf5_file, dataset_folders, action_gap, num_points_to_sample, hand_ahead=0,min_bound=None, max_bound=None, last_mean=100):
    global R_delta_init

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_vis = o3d.geometry.PointCloud()  # Empty point cloud for starters
    firstfirst = True

    with h5py.File(output_hdf5_file, 'w') as output_hdf5:
        output_data_group = output_hdf5.create_group('data')

        demo_index = 0
        total_frames = 0
        mean_init_arm = []
        mean_init_hand = []

        for fid, dataset_folder in enumerate(dataset_folders):
            demos = os.listdir(dataset_folder)
            if 'demo_0' in demos:
                demos = sorted(demos, key=lambda x: int(x.split('_')[1]))
            else: # each folder is named by timestamp: yy-mm-dd_hh-mm-ss
                demos = sorted(demos, key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))

            for demo in demos:

                demo_path = os.path.join(dataset_folder, demo)

                pointcloud = []
                arm_joints = []
                hand_joints = []

                frames = os.listdir(demo_path)
                frames = sorted(frames, key=lambda x: int(x.split('_')[1]))

                clip_length = len(frames)

                for frame_id, frame in enumerate(frames):
                    # if (frame_id < 40 or frame_id > clip_length - 60) and fid >= last_mean:
                    #     continue
                    pointcloud_path = os.path.join(dataset_folder, demo, frame, "point_cloud.ply")
                    arm_joints_path = os.path.join(dataset_folder, demo, frame, "arm_joints.txt")
                    hand_joints_path = os.path.join(dataset_folder, demo, frame, "hand_joints.txt")
                    #wrist_pos_path = os.path.join(dataset_folder, demo, frame, "wrist_pos.txt")
                    #wrist_orn_path = os.path.join(dataset_folder, demo, frame, "wrist_orn.txt")

                    ply = o3d.io.read_point_cloud(pointcloud_path)
                    points = np.asarray(ply.points)
                    colors = np.asarray(ply.colors)
                    point_cloud_data = np.hstack((points, colors))
                    # crop point cloud
                    if min_bound is not None and max_bound is not None:
                        point_cloud_data = crop_pcd(point_cloud_data, min_bound, max_bound)
                    # Check the number of points in the dataset
                    if point_cloud_data.shape[0] < num_points_to_sample:
                        # Calculate how many times to repeat the dataset
                        repeat_count = num_points_to_sample // point_cloud_data.shape[0] + 1
                        extended_point_cloud_data = np.tile(point_cloud_data, (repeat_count, 1))
                        # Now trim the extended dataset to exactly 10000 points if it exceeds
                        point_cloud_data = extended_point_cloud_data[:num_points_to_sample]
                    else:
                        # Randomly select 10000 points if there are enough
                        indices = np.random.choice(point_cloud_data.shape[0], num_points_to_sample, replace=False)
                        point_cloud_data = point_cloud_data[indices]

                    arm_joints_data = np.loadtxt(arm_joints_path)
                    hand_joints_data = np.loadtxt(hand_joints_path)
                    #wrist_pos_data = np.loadtxt(wrist_pos_path)
                    #wrist_orn_data = np.loadtxt(wrist_orn_path)

                    pointcloud.append(copy.deepcopy(point_cloud_data))
                    arm_joints.append(copy.deepcopy(arm_joints_data))
                    hand_joints.append(copy.deepcopy(hand_joints_data))
                    # print("arm:", arm_joints_data)
                    # print("hand:", hand_joints_data)
                    #wrist_poses.append(copy.deepcopy(wrist_pos_data))
                    #wrist_ornes.append(copy.deepcopy(wrist_orn_data))

                    # update pointcloud visualization
                    pcd_vis.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
                    pcd_vis.colors = o3d.utility.Vector3dVector(point_cloud_data[:, 3:])

                    if firstfirst:
                        vis.add_geometry(pcd_vis)
                        firstfirst = False
                    else:
                        vis.update_geometry(pcd_vis)
                    vis.poll_events()
                    vis.update_renderer()

                # Change timing here
                if hand_ahead > 0:
                    pointcloud = np.array(pointcloud)[:-hand_ahead]
                    arm_joints = np.array(arm_joints)[:-hand_ahead]
                    hand_joints = np.array(hand_joints)[hand_ahead:]
                    #wrist_poses = np.array(wrist_poses)[:-hand_ahead]
                    #wrist_ornes = np.array(wrist_ornes)[:-hand_ahead]
                else:
                    pointcloud = np.array(pointcloud)
                    arm_joints = np.array(arm_joints)
                    hand_joints = np.array(hand_joints)
                    #wrist_poses = np.array(wrist_poses)
                    #wrist_ornes = np.array(wrist_ornes)

                actions_arm = np.concatenate((arm_joints[action_gap:], arm_joints[-1:].repeat(action_gap, axis=0)), axis=0)
                #actions_wrist_pos = np.concatenate((wrist_poses[action_gap:], wrist_poses[-1:].repeat(action_gap, axis=0)), axis=0)
                #actions_wrist_orn = np.concatenate((wrist_ornes[action_gap:], wrist_ornes[-1:].repeat(action_gap, axis=0)), axis=0)
                actions_hand = np.concatenate((hand_joints[action_gap:], hand_joints[-1:].repeat(action_gap, axis=0)), axis=0)

                if actions_hand.ndim == 1:
                    actions_hand = np.expand_dims(actions_hand, axis=1)
                    hand_joints = np.expand_dims(hand_joints, axis=1)
                actions = np.concatenate((actions_arm, actions_hand), axis=-1) # merge arm and hand actions
                #actions2 = np.concatenate((actions_wrist_pos, actions_wrist_orn, actions_hand), axis=-1) # merge wrist position and orientation actions

                for j in range(action_gap): # Based on the action_gap, generate the trajectories
                    demo_name = 'demo_{}'.format(demo_index)
                    output_demo_group = output_data_group.create_group(demo_name)
                    print("{} saved".format(demo_name))
                    demo_index += 1

                    output_obs_group = output_demo_group.create_group('obs')
                    output_obs_group.create_dataset('pointcloud', data=np.array(pointcloud)[j::action_gap])
                    output_obs_group.create_dataset('robot0_arm_joints', data=copy.deepcopy(arm_joints)[j::action_gap])
                    output_obs_group.create_dataset('robot0_hand_joints', data=copy.deepcopy(hand_joints)[j::action_gap])
                    #output_obs_group.create_dataset('robot0_eef_pos', data=copy.deepcopy(wrist_poses)[j::action_gap])
                    #output_obs_group.create_dataset('robot0_eef_quat', data=copy.deepcopy(wrist_ornes)[j::action_gap])

                    output_demo_group.create_dataset('actions', data=copy.deepcopy(actions)[j::action_gap])
                    #output_demo_group.create_dataset('actions2', data=copy.deepcopy(actions2)[j::action_gap])

                    # Create 'dones', 'rewards', and 'states'
                    dones = np.zeros(clip_length, dtype=np.int64)
                    dones[-1] = 1  # Set last frame's 'done' to 1
                    output_demo_group.create_dataset('dones', data=dones[j::action_gap])

                    rewards = np.zeros(clip_length, dtype=np.float64)
                    output_demo_group.create_dataset('rewards', data=rewards[j::action_gap])
                    states = np.zeros(clip_length, dtype=np.float64)
                    output_demo_group.create_dataset('states', data=states[j::action_gap])

                    output_demo_group.attrs['num_samples'] = len(actions[j::action_gap])

                    total_frames += len(actions[j::action_gap])

                    if fid < last_mean:
                        mean_init_arm.append(copy.deepcopy(arm_joints[j]))
                        mean_init_hand.append(copy.deepcopy(hand_joints[j]))
                    #mean_init_wrist_pos.append(copy.deepcopy(wrist_poses[j]))
                    #mean_init_wrist_orn.append(copy.deepcopy(wrist_ornes[j]))

        output_data_group.attrs['total'] = total_frames

        # calculate the mean of the initial starting position
        mean_init_arm = np.array(mean_init_arm).mean(axis=0)
        mean_init_hand = np.array(mean_init_hand).mean(axis=0)
        # mean_init_wrist_pos = np.array(mean_init_wrist_pos).mean(axis=0)
        # mean_init_wrist_orn = mean_init_wrist_orn[0]
        output_data_group.attrs['mean_init_arm'] = mean_init_arm
        output_data_group.attrs['mean_init_hand'] = mean_init_hand
        #output_data_group.attrs['mean_init_pos'] = mean_init_wrist_pos
        #output_data_group.attrs['mean_init_quat'] = mean_init_wrist_orn

def process_hdf5_arcap_multi_reprojection(output_hdf5_file, dataset_folders, action_gap, camera_poses, look_ats, up_vecs, fovs, skipframe=0):
    global R_delta_init

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_vis = o3d.geometry.PointCloud()  # Empty point cloud for starters
    firstfirst = True

    with h5py.File(output_hdf5_file, 'w') as output_hdf5:
        output_data_group = output_hdf5.create_group('data')

        demo_index = 0
        total_frames = 0
        mean_init_arm = []
        mean_init_hand = []
        mean_init_wrist_pos = []
        mean_init_wrist_orn = []
        for dataset_folder in dataset_folders:
            demos = os.listdir(dataset_folder)
            demos = sorted(demos, key=lambda x: int(x.split('_')[1]))

            for demo in demos:

                demo_path = os.path.join(dataset_folder, demo)

                rgbs = []
                depths = []
                arm_joints = []
                hand_joints = []
                wrist_poses = []
                wrist_ornes = []

                frames = os.listdir(demo_path)
                frames = sorted(frames, key=lambda x: int(x.split('_')[1]))

                clip_length = len(frames)

                for i,frame in enumerate(frames):
                    if i < skipframe:
                        continue

                    pointcloud_path = os.path.join(dataset_folder, demo, frame, "point_cloud.ply")
                    arm_joints_path = os.path.join(dataset_folder, demo, frame, "arm_joints.txt")
                    hand_joints_path = os.path.join(dataset_folder, demo, frame, "hand_joints.txt")
                    wrist_pos_path = os.path.join(dataset_folder, demo, frame, "wrist_pos.txt")
                    wrist_orn_path = os.path.join(dataset_folder, demo, frame, "wrist_orn.txt")

                    ply = o3d.io.read_point_cloud(pointcloud_path)
                    points = np.asarray(ply.points)
                    colors = np.asarray(ply.colors)
                    point_cloud_data = np.hstack((points, colors))

                    # # Check the number of points in the dataset
                    # if point_cloud_data.shape[0] < num_points_to_sample:
                    #     # Calculate how many times to repeat the dataset
                    #     repeat_count = num_points_to_sample // point_cloud_data.shape[0] + 1
                    #     extended_point_cloud_data = np.tile(point_cloud_data, (repeat_count, 1))
                    #     # Now trim the extended dataset to exactly 10000 points if it exceeds
                    #     point_cloud_data = extended_point_cloud_data[:num_points_to_sample]
                    # else:
                    #     # Randomly select 10000 points if there are enough
                    #     indices = np.random.choice(point_cloud_data.shape[0], num_points_to_sample, replace=False)
                    #     point_cloud_data = point_cloud_data[indices]
                    rgb_images, depth_images = [], []
                    for cid in range(len(camera_poses)):
                        rgb_image, depth_image = render_point_cloud(point_cloud_data[::2] * np.array([1,1,1,255,255,255]), camera_poses[cid], look_ats[cid], up_vecs[cid], fovs[cid],
                                                                    image_width=128, image_height=128)
                        rgb_images.append(rgb_image)
                        depth_images.append(depth_image)

                    arm_joints_data = np.loadtxt(arm_joints_path)
                    hand_joints_data = np.loadtxt(hand_joints_path)
                    wrist_pos_data = np.loadtxt(wrist_pos_path)
                    wrist_orn_data = np.loadtxt(wrist_orn_path)

                    rgbs.append(copy.deepcopy(rgb_images))
                    depths.append(copy.deepcopy(depth_images))
                    arm_joints.append(copy.deepcopy(arm_joints_data))
                    hand_joints.append(copy.deepcopy(hand_joints_data))
                    wrist_poses.append(copy.deepcopy(wrist_pos_data))
                    wrist_ornes.append(copy.deepcopy(wrist_orn_data))

                    # update pointcloud visualization
                    pcd_vis.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
                    pcd_vis.colors = o3d.utility.Vector3dVector(point_cloud_data[:, 3:])
                    for cid in range(len(camera_poses)):
                        cv.imshow(f"rgb{cid}", cv.cvtColor(rgb_images[cid], cv.COLOR_BGR2RGB))
                    if firstfirst:
                        vis.add_geometry(pcd_vis)
                        firstfirst = False
                    else:
                        vis.update_geometry(pcd_vis)
                    vis.poll_events()
                    vis.update_renderer()
                    cv.waitKey(1)

                rgbs = np.array(rgbs) # (num_frames, num_cams, H, W, 3)
                depths = np.array(depths) # (num_frames, num_cams, H, W)
                arm_joints = np.array(arm_joints)
                hand_joints = np.array(hand_joints)
                wrist_poses = np.array(wrist_poses)
                wrist_ornes = np.array(wrist_ornes)

                actions_arm = np.concatenate((arm_joints[action_gap:], arm_joints[-1:].repeat(action_gap, axis=0)), axis=0)
                actions_wrist_pos = np.concatenate((wrist_poses[action_gap:], wrist_poses[-1:].repeat(action_gap, axis=0)), axis=0)
                actions_wrist_orn = np.concatenate((wrist_ornes[action_gap:], wrist_ornes[-1:].repeat(action_gap, axis=0)), axis=0)
                actions_hand = np.concatenate((hand_joints[action_gap:], hand_joints[-1:].repeat(action_gap, axis=0)), axis=0)

                if actions_hand.ndim == 1:
                    actions_hand = np.expand_dims(actions_hand, axis=1)
                    hand_joints = np.expand_dims(hand_joints, axis=1)

                actions = np.concatenate((actions_arm, actions_hand), axis=-1) # merge arm and hand actions
                actions2 = np.concatenate((actions_wrist_pos, actions_wrist_orn, actions_hand), axis=-1) # merge wrist position and orientation actions

                for j in range(action_gap): # Based on the action_gap, generate the trajectories
                    demo_name = 'demo_{}'.format(demo_index)
                    output_demo_group = output_data_group.create_group(demo_name)
                    print("{} saved".format(demo_name))
                    demo_index += 1

                    output_obs_group = output_demo_group.create_group('obs')
                    for cid in range(len(camera_poses)):
                        output_obs_group.create_dataset(f'robot{cid}_rgb_image', data=np.array(rgbs)[:, cid][j::action_gap])
                        output_obs_group.create_dataset(f'robot{cid}_depth_image', data=np.array(depths)[:, cid][j::action_gap])
                    output_obs_group.create_dataset('robot0_arm_joints', data=copy.deepcopy(arm_joints)[j::action_gap])
                    output_obs_group.create_dataset('robot0_hand_joints', data=copy.deepcopy(hand_joints)[j::action_gap])
                    output_obs_group.create_dataset('robot0_eef_pos', data=copy.deepcopy(wrist_poses)[j::action_gap])
                    output_obs_group.create_dataset('robot0_eef_quat', data=copy.deepcopy(wrist_ornes)[j::action_gap])

                    output_demo_group.create_dataset('actions', data=copy.deepcopy(actions)[j::action_gap])
                    output_demo_group.create_dataset('actions2', data=copy.deepcopy(actions2)[j::action_gap])

                    # Create 'dones', 'rewards', and 'states'
                    dones = np.zeros(clip_length, dtype=np.int64)
                    dones[-1] = 1  # Set last frame's 'done' to 1
                    output_demo_group.create_dataset('dones', data=dones[j::action_gap])

                    rewards = np.zeros(clip_length, dtype=np.float64)
                    output_demo_group.create_dataset('rewards', data=rewards[j::action_gap])
                    states = np.zeros(clip_length, dtype=np.float64)
                    output_demo_group.create_dataset('states', data=states[j::action_gap])

                    output_demo_group.attrs['num_samples'] = len(actions[j::action_gap])

                    total_frames += len(actions[j::action_gap])

                    mean_init_arm.append(copy.deepcopy(arm_joints[j]))
                    mean_init_hand.append(copy.deepcopy(hand_joints[j]))
                    mean_init_wrist_pos.append(copy.deepcopy(wrist_poses[j]))
                    mean_init_wrist_orn.append(copy.deepcopy(wrist_ornes[j]))

        output_data_group.attrs['total'] = total_frames

        # calculate the mean of the initial starting position
        mean_init_arm = np.array(mean_init_arm).mean(axis=0)
        mean_init_hand = np.array(mean_init_hand).mean(axis=0)
        mean_init_wrist_pos = np.array(mean_init_wrist_pos).mean(axis=0)
        mean_init_wrist_orn = mean_init_wrist_orn[0]
        output_data_group.attrs['mean_init_arm'] = mean_init_arm
        output_data_group.attrs['mean_init_hand'] = mean_init_hand
        output_data_group.attrs['mean_init_pos'] = mean_init_wrist_pos
        output_data_group.attrs['mean_init_quat'] = mean_init_wrist_orn