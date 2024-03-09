from pybullet_ik_bimanual import LeapPybulletIK
from dataset_utils import *

dataset_base_dir = '/media/jeremy/cde0dfff-70f1-4c1c-82aa-e0d469c14c62/franka_leap_control_datasets/save_wipe_1-14'
sub_dirs = [os.path.join(dataset_base_dir, d) for d in os.listdir(dataset_base_dir) if os.path.isdir(os.path.join(dataset_base_dir, d))]
dataset_folders = sorted(sub_dirs, key=lambda d: extract_dataset_folder_last_two_digits(os.path.basename(d)))

action_gap = 5
num_points_to_sample = 10000

output_hdf5_file = '/media/jeremy/cde0dfff-70f1-4c1c-82aa-e0d469c14c62/franka_leap_control_datasets/hand_wiping_1-14_{}actiongap_{}points.hdf5'.format(action_gap, num_points_to_sample)

process_hdf5(output_hdf5_file, dataset_folders, action_gap, num_points_to_sample)
