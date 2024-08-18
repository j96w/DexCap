import os
import shutil
import cv2
import pyrealsense2 as rs

def save_pose(file_path, translation, rotation):
    with open(file_path, 'w') as f:
        f.write(' '.join(map(str, translation)) + ' ' + ' '.join(map(str, rotation)))

def save_image(file_path, image):
    cv2.imwrite(file_path, image)

def setup_folder_structure(dataset_folder):
    if os.path.exists(dataset_folder):
        overwrite = input(f"Folder {dataset_folder} already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() == 'y':
            shutil.rmtree(dataset_folder)
        else:
            print("Operation cancelled.")
            exit()
    os.makedirs(dataset_folder)
    data_folder = os.path.join(dataset_folder, "data")
    os.makedirs(data_folder)
    return dataset_folder, data_folder

def save_camera_intrinsics(dataset_folder, intrinsics):
    camera_matrix_path = os.path.join(dataset_folder, "camera_matrix.txt")
    with open(camera_matrix_path, 'w') as f:
        f.write(f"fx {intrinsics.fx}\n")
        f.write(f"fy {intrinsics.fy}\n")
        f.write(f"ppx {intrinsics.ppx}\n")
        f.write(f"ppy {intrinsics.ppy}\n")
        f.write(f"width {intrinsics.width}\n")
        f.write(f"height {intrinsics.height}\n")

def configure_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    pipeline_profile = pipeline.start(config)
    return pipeline, pipeline_profile
