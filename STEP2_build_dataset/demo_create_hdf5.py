from dataset_utils import *

# dataset_base_dir = '[PATH_TO_RAW_DATA_DOWNLOAD]/save_wipe_1-14'
dataset_base_dir = '[PATH_TO_RAW_DATA_DOWNLOAD]/save_packaging_wild_1-20'

sub_dirs = [os.path.join(dataset_base_dir, d) for d in os.listdir(dataset_base_dir) if os.path.isdir(os.path.join(dataset_base_dir, d))]
dataset_folders = sorted(sub_dirs, key=lambda d: extract_dataset_folder_last_two_digits(os.path.basename(d)))

action_gap = 5
num_points_to_sample = 10000

# output_hdf5_file = '[PATH_TO_SAVE_FOLDER]/hand_wiping_1-14_{}actiongap_{}points.hdf5'.format(action_gap, num_points_to_sample)
output_hdf5_file = '[PATH_TO_SAVE_FOLDER]/hand_packaging_wild_1-20_{}actiongap_{}points.hdf5'.format(action_gap, num_points_to_sample)

# process_hdf5(output_hdf5_file, dataset_folders, action_gap, num_points_to_sample)
process_hdf5(output_hdf5_file, dataset_folders, action_gap, num_points_to_sample, in_wild_data=True)