import numpy as np

# Realsense L515 camera
between_cam = np.eye(4)
between_cam[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
between_cam[:3, 3] = np.array([0.0, 0.76, 0.0])

between_cam_2 = np.eye(4)
between_cam_2[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
between_cam_2[:3, 3] = np.array([0.0, -0.032, 0.0])

between_cam_3 = np.eye(4)
between_cam_3[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
between_cam_3[:3, 3] = np.array([0.0, -0.064, 0.0])