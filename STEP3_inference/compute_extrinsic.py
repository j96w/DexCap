import numpy as np
import torch
import open3d as o3d

def optimal_transformation_batch(S1, S2):
    """
    S1: [num_envs, num_points, 3]
    S2: [num_envs, num_points, 3]
    weight: [num_envs, num_points]
    """
    S1 = torch.from_numpy(S1)
    S2 = torch.from_numpy(S2)
    c1 = S1.mean(dim=0)
    c2 = S2.mean(dim=0)
    H = (S1 - c1).transpose(0,1) @ (S2 - c2)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.mH
    R = V @ U.transpose(0,1)
    if R.det() < 0:
        V[:,-1] *= -1
        R = V @ U.transpose(0,1)
    t = c2 - R@c1
    return R, t

calib_pcd = o3d.io.read_point_cloud("configs/raw_point_cloud.ply")

calib_points_id = [264092, 272708, 203054, 208258]

all_calib_points = np.asarray(calib_pcd.points)
calib_points = all_calib_points[calib_points_id]

ref_points = np.array([[0, 0, 0], [0.2775, 0, 0], [0, 0.2155, 0], [0.2775, 0.2155, 0]])

R,t = optimal_transformation_batch(calib_points, ref_points)
R = R.numpy()
t = t.numpy()


length = np.linalg.norm(calib_points[1] - calib_points[0])
for i in range(len(calib_points)):
    print((R@calib_points[i]) + t-ref_points[i])
calib_pcd.rotate(R, center=(0.0, 0.0, 0.0))
calib_pcd.translate(t)
o3d.visualization.draw_geometries_with_vertex_selection([calib_pcd])
np.savez(f"configs/tf_camera.npz", R=R, t=t)