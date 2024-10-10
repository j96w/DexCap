from dataset_utils import *

dataset_folders = ['/home/ericcsr/robot_imitation/data/2024-10-05-15-09-14',
                   '/home/ericcsr/robot_imitation/data/2024-10-05-15-34-47',
                   '/home/ericcsr/robot_imitation/data/2024-10-05-16-04-44',
                   '/home/ericcsr/robot_imitation/data/2024-10-05-16-32-14',
                   '/home/ericcsr/robot_imitation/data/2024-10-05-16-59-32',
                   '/home/ericcsr/robot_imitation/data/2024-10-07-16-35-23',
                   '/home/ericcsr/robot_imitation/data/2024-10-07-16-25-37']

action_gap = 2
point_cloud_data = 10000 # Greater than 1000

output_hdf5_file = f"bottle_stage12_v3_{action_gap}gap.hdf5"

#max_bound = np.array([1, 1, 0.5])
#min_bound = np.array([-1, -1, 0.018])

process_hdf5_arcap_multi(output_hdf5_file, dataset_folders, action_gap, point_cloud_data, hand_ahead=0, last_mean=5)