# ffmpeg -framerate 10 -i saved_current_frames/%d.jpg -c:v mpeg4 -pix_fmt yuv420p saved_run_video.mp4

import argparse
import h5py
import time
import traceback
import numpy as np
import redis
import torch
import pickle
from transforms3d.euler import quat2mat
import pybullet as pb

from glob import glob
from scipy.spatial.transform import Rotation
from gprs.franka_interface import FrankaInterface
# from gprs.utils.io_devices import SpaceMouse
# from gprs.utils.input_utils import input2action
from gprs.utils import YamlConfig

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS
import matplotlib.pyplot as plt
from realsense_module import DepthCameraModule, crop_pcd
# from deoxys_spacemouse.input_utils import input2action
# from deoxys_spacemouse.spacemouse import SpaceMouse

camera_poses = [np.array([-0.4, -0.25, 0.3]),np.array([-0.6, -0.25, 0.3]), np.array([-0.2, -0.25, 0.3])]
look_ats = [np.array([-0.4, 0.2, 0.1]), np.array([-0.4, 0.2, 0.1]), np.array([-0.4, 0.2, 0.1])]
up_dirs = [np.array([0.0, 0.0, 1.0]),np.array([0.0, 0.0, 1.0]),np.array([0.0, 0.0, 1.0])]
fovs = [65, 65, 65]

pb.connect(pb.GUI)

def convert_to_hardware(joint_angles):
    real_right_robot_hand_q = np.zeros(16)
    real_right_robot_hand_q[0:16] = joint_angles[0:16]
    real_right_robot_hand_q[0:2] = real_right_robot_hand_q[0:2][::-1]
    real_right_robot_hand_q[4:6] = real_right_robot_hand_q[4:6][::-1]
    real_right_robot_hand_q[8:10] = real_right_robot_hand_q[8:10][::-1]
    real_right_robot_hand_q[:16] += np.pi
    return real_right_robot_hand_q

def reverse_conversion(joint_angles):
    real_right_robot_hand_q = np.zeros(16)
    real_right_robot_hand_q[0:16] = joint_angles[0:16]
    real_right_robot_hand_q[0:2] = real_right_robot_hand_q[0:2][::-1]
    real_right_robot_hand_q[4:6] = real_right_robot_hand_q[4:6][::-1]
    real_right_robot_hand_q[8:10] = real_right_robot_hand_q[8:10][::-1]
    real_right_robot_hand_q[:16] -= np.pi
    return real_right_robot_hand_q


class RobotEnv:
    def __init__(self, hdf5_file, rs_module, head_pos, head_orn, obs_horizon = 5, handedness = "right"):
        if handedness in ["right", "both"]:
            self.redis = redis.Redis(host='localhost', port=6669, db=0)
        self.rs_module = rs_module
        self.hdf5_file = hdf5_file
        self.head_pos = head_pos
        self.head_orn = head_orn
        self.obs_horizon = obs_horizon
        self.load_init_from_hdf5()
        self.handedness = handedness
        #self.pcd_idx = np.random.choice(10000, 10000, replace=False)

        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self.handedness = handedness
        left_config_file = "robot_config/alice_left.yml"
        right_config_file = "robot_config/alice.yml"
        if handedness == "left" or handedness == "both":
            self.left_robot_interface = FrankaInterface(
                left_config_file, use_visualizer=False, has_gripper=True
            )
        if handedness == "right" or handedness == "both":
            self.right_robot_interface = FrankaInterface(
                right_config_file, use_visualizer=False, has_gripper=False
            )
        self.pcd_buffer = []
        self.arm_q_buffer = []
        self.hand_q_buffer = []

        self.reset_cnt = 0

    def init_robot(self, init_arm_q=None, init_hand_q=None, handedness= "right"):
        robot_interface = self.right_robot_interface if handedness == "right" else self.left_robot_interface
        self.controller_cfg = YamlConfig("robot_config/joint-impedance-controller.yml").as_easydict()
        robot_interface._state_buffer = []


        # first reset the arm to a initial pose
        fixed_joints = [
            0.0,
            -0.49826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.00396583422025,
            0.8480939705504309
        ]
        a = fixed_joints
        if handedness == "right":
            paper_q = np.array([0.0 for _ in range(16)])
        else:
            a = fixed_joints + [-1]
        if handedness == "right":
            initial_hand_q = pickle.loads(self.redis.get("right_leap_joints"))
        for i in range(10):
            robot_interface.control(
                control_type="JOINT_POSITION",
                action=a,
                mode=0.0,
                controller_cfg=self.controller_cfg,
            )
            time.sleep(0.5)
            if handedness == "right":
                self.redis.set('right_leap_action', pickle.dumps(i/10*convert_to_hardware(paper_q)+(10-i)/10*initial_hand_q))
        if handedness == "right":
            input("Press Enter to continue...")
        # first reset the arm to the mean init state in the hdf5 dataset
        for i in range(50):
            a = init_arm_q
            if handedness == "left":
                a = np.hstack([init_arm_q, init_hand_q])
            robot_interface.control(
                control_type="JOINT_POSITION",
                action=a,
                mode=0.0,
                controller_cfg=self.controller_cfg,
            )
            # Hand joint angle may need conversion... hand install is inversed...
            if handedness == "right":
                self.redis.set('right_leap_action', pickle.dumps(convert_to_hardware(i/50*init_hand_q+(50-i)/50*paper_q)))
            time.sleep(0.1)
        if handedness == "left":
            self.last_gripper_status = init_hand_q[0]

    def load_init_from_hdf5(self):
        with h5py.File(self.hdf5_file, 'r') as file:
            data_group = file['data']
            self.init_arm = data_group.attrs['mean_init_arm']
            self.init_hand = data_group.attrs['mean_init_hand']

    def get_robot_states(self):
        # not implemented
        if self.handedness == "right" or self.handedness == "both":
            right_arm_joints = np.array(self.right_robot_interface.last_state.q)
            right_hand_joints = reverse_conversion(pickle.loads(self.redis.get("right_leap_joints")))
        if self.handedness == "left" or self.handedness == "both":
            left_arm_joints = np.array(self.left_robot_interface.last_state.q)
            left_hand_joints = self.last_gripper_status


        if self.handedness == "right":
            robot0_arm_joints = right_arm_joints
            robot0_hand_joints = right_hand_joints
        elif self.handedness == "left":
            robot0_arm_joints = left_arm_joints
            robot0_hand_joints = left_hand_joints
        else:
            robot0_arm_joints = np.hstack([left_arm_joints, right_arm_joints])
            robot0_hand_joints = np.hstack([left_hand_joints, right_hand_joints])
        print("arm state:", robot0_arm_joints)
        return robot0_arm_joints, robot0_hand_joints

    def get_point_cloud(self, keep_original=False):
        pointcloud = self.rs_module.receive()
        pointcloud = self.rs_module.to_world(pointcloud, self.head_pos, Rotation.from_quat(self.head_orn))
        if self.handedness == "right":
            cropped = crop_pcd(pointcloud, (-0.45,0.1, 0.015), (0.,0.6,0.5))
        elif self.handedness == "left":
            cropped = crop_pcd(pointcloud, (0.0,0.1, -0.005), (0.7,0.3,0.5))
        else:
            cropped = crop_pcd(pointcloud, (-0.7,0.1, 0.04), (0.,0.6,0.5))
        
        # Subsample to 500 points
        if not keep_original:
            if cropped.shape[0] < 10000:
                # Calculate how many times to repeat the dataset
                repeat_count = 10000 // cropped.shape[0] + 1
                extended_cropped = np.tile(cropped, (repeat_count, 1))
                # Now trim the extended dataset to exactly 10000 points if it exceeds
                cropped = extended_cropped[:10000]
            else:
                # Randomly select 10000 points if there are enough
                indices = np.random.choice(cropped.shape[0], 10000, replace=False)
                cropped = cropped[indices]
        self.rs_module.visualize_pcd(cropped)
        return cropped
        
        
    def get_state(self, first=False):
        robot0_arm_joints, robot0_hand_joints = self.get_robot_states()
        pointcloud = self.get_point_cloud(keep_original=False)
        pointcloud = TensorUtils.to_device(torch.FloatTensor(pointcloud), self.device)
        robot0_arm_joints = TensorUtils.to_device(torch.FloatTensor(robot0_arm_joints), self.device)
        robot0_hand_joints = TensorUtils.to_device(torch.FloatTensor(robot0_hand_joints), self.device)
        self.pcd_buffer.append(pointcloud)
        self.arm_q_buffer.append(robot0_arm_joints)
        self.hand_q_buffer.append(robot0_hand_joints)

        if len(self.pcd_buffer) > self.obs_horizon:
            self.pcd_buffer.pop(0)
            self.arm_q_buffer.pop(0)
            self.hand_q_buffer.pop(0)
        else:
            self.pcd_buffer = [pointcloud for _ in range(self.obs_horizon)]
            self.arm_q_buffer = [robot0_arm_joints for _ in range(self.obs_horizon)]
            self.hand_q_buffer = [robot0_hand_joints for _ in range(self.obs_horizon)]
        

        return_state = {
            'pointcloud': torch.stack(self.pcd_buffer, dim=0),
            'robot0_arm_joints': torch.stack(self.arm_q_buffer, dim=0),
            'robot0_hand_joints': torch.stack(self.hand_q_buffer, dim=0)
        }
        print("Here")

        return return_state

    def check_arrived(self, goal_arm, goal_hand):
        # threshold = 0.05
        # threshold_hand = 0.3

        # robot0_arm_joints, robot0_hand_joints = self.get_robot_states()

        # diff_arm = np.abs(goal_arm - robot0_arm_joints)
        # diff_hand = np.abs(goal_hand - robot0_hand_joints)
        # print("Tracking error:", diff_arm, diff_hand)

        #return np.all(diff_arm <= threshold) #and np.all(diff_hand <= threshold_hand)
        if self.reset_cnt > 2:
            self.reset_cnt = 0
            return True
        else:
            self.reset_cnt += 1
            return False
        #return True
            
def run_trained_agent(args):
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)

    # device
    device = torch.device("cpu")

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    policy.start_episode()

    # create environment
    rs_module = DepthCameraModule(manual_visualize=True)
    depth_tf = np.load("configs/tf_camera.npz")
    depth_R = Rotation.from_matrix(depth_tf["R"])
    depth_t = depth_tf["t"]
    quest_tf = np.load("configs/calibration.npz")
    quest_R = Rotation.from_quat(quest_tf["rel_rot"])
    quest_t = quest_tf["rel_pos"]
    delta_orn = quest_R.inv() * depth_R
    delta_pos = quest_R.inv().apply(depth_t - quest_t)
    rs_module.load_world(delta_pos, delta_orn)
    head_tf = np.load("configs/calibration_head.npz")
    head_pos, head_orn = head_tf["rel_pos"], head_tf["rel_rot"]

    env = RobotEnv(args.dataset_path, rs_module, 
                    head_pos=head_pos, head_orn=head_orn, 
                    handedness=args.handedness, obs_horizon=args.obs_horizon)
    
    if args.handedness == "right":
        env.init_robot(init_hand_q=env.init_hand, init_arm_q=env.init_arm, handedness="right")
    elif args.handedness == "left":
        env.init_robot(init_hand_q=env.init_hand, init_arm_q=env.init_arm, handedness="left")
    else:
        #breakpoint()
        env.init_robot(init_hand_q=env.init_hand[:1], init_arm_q=env.init_arm[:7], handedness="left")
        env.init_robot(init_hand_q=env.init_hand[1:], init_arm_q=env.init_arm[7:], handedness="right")

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    substep = 0
    step = 0
    
    arrived = True
    time_ts = time.time()
    goal_arm = None
    goal_hand = None
    with torch.no_grad():
        while True:
            input_state = env.get_state()

            if arrived:
                #print("Arrived!")
                pred_action = policy(ob=input_state)
                if not args.handedness == "both":
                    goal_arm_model = pred_action[:7]
                    goal_hand_model = pred_action[7:]
                else:
                    goal_arm_model = pred_action[:14]
                    goal_hand_model = pred_action[14:]
                if args.handedness in ["left", "both"]:
                    env.last_gripper_status = goal_hand_model[0]
                goal_arm = goal_arm_model
                goal_hand = goal_hand_model
                arrived = False
                step += 1
            
            goal_arm_ = goal_arm
            a = goal_arm_
            if args.handedness == "left" or args.handedness == "both":
                a = np.hstack([goal_arm_[:7] , 1 if goal_hand[0]>0 else -1])
                env.left_robot_interface.control(
                    control_type="JOINT_IMPEDANCE",
                    action=a,
                    mode=0.0,
                    controller_cfg=env.controller_cfg,
                )
                goal_hand_ = goal_hand[:1].copy()
            if args.handedness == "right" or args.handedness == "both":
                a = goal_arm_[(0 if args.handedness == "right" else 7):]
                env.right_robot_interface.control(
                    control_type="JOINT_IMPEDANCE",
                    action=a,
                    mode=0.0,
                    controller_cfg=env.controller_cfg,
                )
                goal_hand_ = goal_hand[(0 if args.handedness == "right" else 1):].copy()
                goal_hand_ = convert_to_hardware(goal_hand_)
                env.redis.set('right_leap_action', pickle.dumps(goal_hand_))

            
            arrived = env.check_arrived(goal_arm, goal_hand)

            substep += 1
            # print control frequency
            print("Frequency:", 1 / (time.time() - time_ts))
            time_ts = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # Use the dataset to load a mean initial starting pose for the robot
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='xxx.hdf5',
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--handedness",
        type=str,
        default="right"
    )

    parser.add_argument(
        "--obs_horizon",
        type=int,
        default=5
    )
    args = parser.parse_args()

    try:
        run_trained_agent(args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        if args.error_path is not None:
            # write traceback to file
            f = open(args.error_path, "w")
            f.write(res_str)
            f.close()
        raise e