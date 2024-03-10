import argparse
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calibrate and save in dataset folder")
    parser.add_argument("--directory", type=str, default="", help="Directory with saved data")
    parser.add_argument("--default", type=str, default="default_offset", help="Directory with saved data")

    args = parser.parse_args()

    if os.path.exists("{}/calib_offset.txt".format(args.directory)):
        response = (
            input(
                f"calib_offset.txt already exists. Do you want to override? (y/n): "
            )
            .strip()
            .lower()
        )
        if response != "y":
            print("Exiting program without overriding the existing directory.")
            sys.exit()

    default_offset = np.loadtxt(os.path.join(args.default, "calib_offset.txt"))
    default_ori = np.loadtxt(os.path.join(args.default, "calib_ori_offset.txt"))
    default_offset_left = np.loadtxt(os.path.join(args.default, "calib_offset_left.txt"))
    default_ori_left = np.loadtxt(os.path.join(args.default, "calib_ori_offset_left.txt"))

    default_ori_matrix = euler2mat(*default_ori)
    default_ori_matrix_left = euler2mat(*default_ori_left)

    # extract offset and ori offset from directory
    frame_dirs = os.listdir("./test_data")
    list = []
    list_ori = []
    list_left = []
    list_ori_left = []

    for frame_dir in frame_dirs:
        print(frame_dir)
        if "_left" in frame_dir:
            if "_ori" in frame_dir:
                delta_ori_euler = np.loadtxt(os.path.join("./test_data", frame_dir))
                delta_ori_matrix = euler2mat(*delta_ori_euler)
                combined_matrix = np.dot(delta_ori_matrix, default_ori_matrix_left) 
                list_ori_left.append(R.from_matrix(combined_matrix).as_euler('xyz', degrees=False))
            else:
                list_left.append(np.loadtxt(os.path.join("./test_data", frame_dir)))
        else:
            if "_ori" in frame_dir:
                delta_ori_euler = np.loadtxt(os.path.join("./test_data", frame_dir))
                delta_ori_matrix = euler2mat(*delta_ori_euler)
                combined_matrix = np.dot(delta_ori_matrix, default_ori_matrix) 
                list_ori.append(R.from_matrix(combined_matrix).as_euler('xyz', degrees=False))
            else:
                list.append(np.loadtxt(os.path.join("./test_data", frame_dir)))

    # calculate offset
    print("delta offset", np.mean(list, axis=0))
    calib_offset = default_offset + np.mean(list, axis=0)
    print("final calib offset", calib_offset)
    np.savetxt('{}/calib_offset.txt'.format(args.directory), calib_offset, delimiter=',')

    # calculate ori
    print("delta ori offset", np.mean(list_ori, axis=0))
    calib_ori_offset = np.mean(list_ori, axis=0)
    np.savetxt('{}/calib_ori_offset.txt'.format(args.directory), calib_ori_offset, delimiter=',')
    print("final calib ori offset", calib_ori_offset)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # calculate offset
    print("delta offset left", np.mean(list_left, axis=0))
    calib_offset_left = default_offset_left + np.mean(list_left, axis=0)
    print("final calib offset left", calib_offset_left)
    np.savetxt('{}/calib_offset_left.txt'.format(args.directory), calib_offset_left, delimiter=',')

    # calculate ori
    print("delta ori offset", np.mean(list_ori_left, axis=0))
    calib_ori_offset_left = np.mean(list_ori_left, axis=0)
    np.savetxt('{}/calib_ori_offset_left.txt'.format(args.directory), calib_ori_offset_left, delimiter=',')
    print("final calib ori offset left", calib_ori_offset_left)