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
import matplotlib.pyplot as plt
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
    def __init__(self, init_arm, init_hand, handedness="right"):
        if handedness in ["right", "both"]:
            self.redis = redis.Redis(host='localhost', port=6669, db=0)
        self.init_arm = init_arm
        self.init_hand = init_hand
        #self.pcd_idx = np.random.choice(10000, 10000, replace=False)
        self.handedness = handedness
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
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


        # self.REALROBOT_RIGHT_HAND_OFFSET_CONFIG_PATH = "./config/realrobot_right_hand_offset.yml"
        # self.REALROBOT_RIGHT_HAND_OFFSET = None
        # with open(self.REALROBOT_RIGHT_HAND_OFFSET_CONFIG_PATH) as f:
        #     self.REALROBOT_RIGHT_HAND_OFFSET = yaml.safe_load(f)
        self.reset_cnt = 0

    def init_robot(self, init_hand_q, init_arm_q, handedness = "right"):
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

    def get_robot_states(self):
        # not implemented
        if self.handedness == "right" or self.handedness == "both":
            right_arm_joints = np.array(self.right_robot_interface.last_state.q)
            right_hand_joints = reverse_conversion(pickle.loads(self.redis.get("right_leap_joints")))
        if self.handedness == "left" or self.handedness == "both":
            left_arm_joints = np.array(self.left_robot_interface.last_state.q)
            left_hand_joints = None


        if self.handedness == "right":
            robot0_arm_joints = right_arm_joints
            robot0_hand_joints = right_hand_joints
        elif self.handedness == "left":
            robot0_arm_joints = left_arm_joints
            robot0_hand_joints = left_hand_joints
        else:
            robot0_arm_joints = np.hstack([left_arm_joints, right_arm_joints])
            robot0_hand_joints = right_hand_joints
        
        return robot0_arm_joints, robot0_hand_joints
        
    def get_state(self, first=False):
        return None

    def check_arrived(self, goal_arm, goal_hand):
        # threshold = 0.05
        # threshold_hand = 0.3

        # robot0_arm_joints, robot0_hand_joints = self.get_robot_states()

        # diff_arm = np.abs(goal_arm - robot0_arm_joints)
        # diff_hand = np.abs(goal_hand - robot0_hand_joints)
        # print("Tracking error:", diff_arm, diff_hand)

        #return np.all(diff_arm <= threshold) #and np.all(diff_hand <= threshold_hand)
        # if self.reset_cnt > 0:
        #     self.reset_cnt = 0
        #     return True
        # else:
        #     self.reset_cnt += 1
        #     return False
        return True          

def run_trained_agent(args):
    # load ckpt dict and get algo name for sanity checks

    # device
    device = torch.device("cpu")

    # Load trajectory
    traj = np.load(f"trajs/{args.trajectory}.npz")
    arm_q_traj = traj["arm_qs"][args.start_idx::args.stride] # 3 is harded coded stride.
    wrist_poses = traj["wrist_poses"][args.start_idx::args.stride]
    wrist_orns = traj["wrist_orns"][args.start_idx::args.stride]
    hand_q_traj = traj["hand_qs"][args.start_idx::args.stride]
    # create environment
    env = RobotEnv(arm_q_traj[0], hand_q_traj[0], handedness=args.handedness)
    if args.handedness == "right":
        env.init_robot(env.init_hand, env.init_arm, handedness="right")
    elif args.handedness == "left":
        env.init_robot(env.init_hand, env.init_arm, handedness="left")
    else:
        env.init_robot(env.init_hand[:1], env.init_arm[:7], handedness="left")
        env.init_robot(env.init_hand[1:], env.init_arm[7:], handedness="right")
    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    substep = 0
    step = 0
    
    arrived = True
    time_ts = time.time()
    with torch.no_grad():
        while True:
            if arrived:
                goal_arm = arm_q_traj[step]
                goal_hand = hand_q_traj[step]
                print("play back")
                arrived = False
                step += 1
                if step >= len(arm_q_traj):
                    break
            
            goal_arm_ = goal_arm
            a = goal_arm_
            if args.handedness == "left" or args.handedness == "both":
                a = np.hstack([goal_arm_[:7] , goal_hand[:1]])
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

            arrived = env.check_arrived(goal_arm, goal_hand_)

            substep += 1
            # print control frequency
            print("Frequency:", 1 / (time.time() - time_ts))
            time_ts = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Trajectory to load and replay
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="path to npz file containing trajectory to replay",
    )


    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1
    )

    parser.add_argument(
        "--action_space",
        type=str,
        default="joint")

    parser.add_argument(
        "--handedness", 
        type=str, 
        default="right"
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        default=0
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