import pyrealsense2 as rs
import numpy as np
import open3d as o3d

def crop_pcd(pcd, min_bound, max_bound):
    pcd_pos = pcd[:,:3]
    pcd_color = pcd[:,3:]
    mask = np.logical_and(np.all(pcd_pos > min_bound, axis=1), np.all(pcd_pos < max_bound, axis=1))
    pcd_pos = pcd_pos[mask]
    pcd_color = pcd_color[mask]
    return np.hstack([pcd_pos, pcd_color])

# Depth camera related functions and constants
class DepthCameraModule:
    def __init__(self, is_decimate=False, visualize=False, manual_visualize=False):
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("Camera error or not connected!")
            exit(0)
        
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
        pipeline.start(config)
        depth_sensor = device.query_sensors()[0]
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

        profile = pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height
        pc = rs.pointcloud()
        if is_decimate:
            decimate = rs.decimation_filter()
            decimate.set_option(rs.option.filter_magnitude, 4 ** 1)
        colorizer = rs.colorizer()
        align_to = rs.stream.depth
        self.align = rs.align(align_to)
        self.pipeline = pipeline
        self.pc = pc
        self.is_decimate = is_decimate
        self.colorizer = colorizer
        self.visualize = visualize
        self.manual_visualize = manual_visualize
        self.w, self.h = w, h
        self.first_run = True
        self.all_black_mask = np.load("all_black_mask.npy")
        if self.visualize or self.manual_visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

    def load_world(self, delta_pos, delta_orn):
        self.delta_pos = delta_pos
        self.delta_orn = delta_orn

    def to_world(self, pcd, head_pos, head_orn):
        pcd_pos = pcd[:,:3]
        pcd_color = pcd[:,3:]
        pcd_world = head_orn.apply(self.delta_orn.apply(pcd_pos) + self.delta_pos) + head_pos
        return np.hstack([pcd_world, pcd_color])

    def receive(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        points = self.pc.calculate(depth_frame)
        if self.is_decimate:
            depth_frame = self.decimate.process(depth_frame)
        points = self.pc.calculate(depth_frame)
        self.pc.map_to(color_frame)
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32)
        verts = verts.reshape(-1, 3)[~self.all_black_mask]
        colors = np.asanyarray(color_frame.get_data(), np.float32).reshape(-1,3)[~self.all_black_mask] / 255.0
        vis_verts = verts[::10]
        vis_colors = colors[::10]
        if self.visualize:
            vis_verts = verts[::20]
            vis_colors = colors[::20]
            if self.first_run:
                self.o3d_pcd = o3d.geometry.PointCloud()
                self.o3d_pcd.points = o3d.utility.Vector3dVector(vis_verts)
                self.o3d_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
                coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window()
                self.vis.add_geometry(self.o3d_pcd)
                self.vis.add_geometry(coordframe)
                self.first_run = False
            else:
                self.o3d_pcd.points = o3d.utility.Vector3dVector(vis_verts)
                self.o3d_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
                self.vis.update_geometry(self.o3d_pcd)
                self.vis.poll_events()
                self.vis.update_renderer()
        return np.hstack([vis_verts, vis_colors])

    def visualize_pcd(self, pcd):
        vis_verts = pcd.copy()[:,:3]
        vis_colors = pcd.copy()[:,3:]
        if self.first_run:
            self.o3d_pcd = o3d.geometry.PointCloud()
            self.o3d_pcd.points = o3d.utility.Vector3dVector(vis_verts)
            self.o3d_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
            coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.add_geometry(self.o3d_pcd)
            self.vis.add_geometry(coordframe)
            self.first_run = False
        else:
            self.o3d_pcd.points = o3d.utility.Vector3dVector(vis_verts)
            self.o3d_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
            self.vis.update_geometry(self.o3d_pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        self.pipeline.stop()
        if self.visualize:
            self.vis.destroy_window()

if __name__ == "__main__":
    camera = DepthCameraModule(is_decimate=False, visualize=True)
    while True:
        camera.receive()