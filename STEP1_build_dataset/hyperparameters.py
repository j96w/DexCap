import numpy as np
import open3d as o3d
import yaml

# load calibration results
REALROBOT_RIGHT_HAND_OFFSET_CONFIG_PATH = "./config/realrobot_right_hand_offset.yml"
REALROBOT_RIGHT_HAND_OFFSET = None
with open(REALROBOT_RIGHT_HAND_OFFSET_CONFIG_PATH) as f:
    REALROBOT_RIGHT_HAND_OFFSET = yaml.safe_load(f)

REALROBOT_LEFT_HAND_OFFSET_CONFIG_PATH = "./config/realrobot_left_hand_offset.yml"
REALROBOT_LEFT_HAND_OFFSET = None
with open(REALROBOT_LEFT_HAND_OFFSET_CONFIG_PATH) as f:
    REALROBOT_LEFT_HAND_OFFSET = yaml.safe_load(f)

# chest camera fixed transformation
between_cam = np.eye(4)
between_cam[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0, 0.0, -1.0]])
between_cam[:3, 3] = np.array([0.0, 0.076, 0.0])

between_cam_2 = np.eye(4)
between_cam_2[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]])
between_cam_2[:3, 3] = np.array([0.0, -0.032, 0.0])

between_cam_3 = np.eye(4)
between_cam_3[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]])
between_cam_3[:3, 3] = np.array([0.0, -0.064, 0.0])

# the corner points of the robot table, used for remove redundant points
robot_table_corner_points = np.array([
    [-0.262721, -0.25, -0.077183],
    [0.228490, -0.23, -0.079729],
    [0.390924, -0.215, -0.783124],
    [-0.372577, -0.235, -0.781769]
])

# robot table sweep list
table_sweep_list = [0.020, 0.021, 0.022, 0.023, 0.024, 0.025]

# depth camera intrinsic
o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    1280, 720,
    898.2010498046875,
    897.86669921875,
    657.4981079101562,
    364.30950927734375)