import socket
import time
from argparse import ArgumentParser
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as pb
from rigidbodySento import create_primitive_shape
from ip_config import *
from rokoko_module import RokokoModule
#from realsense_module import DepthCameraModule
from quest_robot_module import QuestRightArmLeapModule

# Robot deployment imports
import redis
import pickle
from gprs.franka_interface import FrankaInterface
from gprs.utils import YamlConfig

def convert_to_hardware(joint_angles):
    real_right_robot_hand_q = np.zeros(16)
    real_right_robot_hand_q[0:16] = joint_angles[0:16]
    real_right_robot_hand_q[0:2] = real_right_robot_hand_q[0:2][::-1]
    real_right_robot_hand_q[4:6] = real_right_robot_hand_q[4:6][::-1]
    real_right_robot_hand_q[8:10] = real_right_robot_hand_q[8:10][::-1]
    real_right_robot_hand_q[:16] += np.pi
    return real_right_robot_hand_q


def init_robot(redis_client, robot_interface):
    hand_target = [0.0 for _ in range(16)]
    redis_client.set('right_leap_action', pickle.dumps(convert_to_hardware(hand_target)))

    controller_cfg = YamlConfig("robot_config/joint-impedance-controller.yml").as_easydict()
    robot_interface._state_buffer = []


    # first reset the arm to a initial pose
    fixed_joints = [
        0.0,
        -0.49826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.00396583422025,
        0.8480939705504309,
    ]
    paper_q = [0.0 for _ in range(16)]
    for _ in range(10):
        robot_interface.control(
            control_type="JOINT_POSITION",
            action=fixed_joints,
            mode=0.0,
            controller_cfg=robot_interface,
        )
        time.sleep(0.5)
        redis_client.set('right_leap_action', pickle.dumps(convert_to_hardware(paper_q)))
    return controller_cfg


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frequency", type=int, default=30)
    args = parser.parse_args()
    c = pb.connect(pb.GUI)
    vis_sp = []
    c_code = c_code = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]]
    for i in range(4):
        vis_sp.append(create_primitive_shape(pb, 0.1, pb.GEOM_SPHERE, [0.02], color=c_code[i]))
    redis_client = redis.Redis(host='localhost',port=6669, db=0)
    robot_interface = FrankaInterface('robot_config/alice.yml', use_visualizer=False, has_gripper=False)
    controller_cfg = init_robot(redis_client, robot_interface)
    #camera = DepthCameraModule(is_decimate=False, visualize=False)
    rokoko = RokokoModule(VR_HOST, HAND_INFO_PORT, ROKOKO_PORT)
    quest = QuestRightArmLeapModule(VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=None)

    start_time = time.time()
    fps_counter = 0
    packet_counter = 0
    print("Initialization completed")
    current_ts = time.time()
    while True:
        now = time.time()
        # TODO: May cause communication issues, need to tune on AR side.
        if now - current_ts < 1 / args.frequency: 
            continue
        else:
            current_ts = now
        try:
            #point_cloud = camera.receive()
            left_positions, right_positions = rokoko.receive()
            rokoko.send_joint_data(np.vstack([left_positions, right_positions]))
            right_wrist, head_pose= quest.receive()
            if right_wrist is not None:
                right_wrist_orn = Rotation.from_quat(right_wrist[1])
                right_wrist_pos = right_wrist[0]
                head_pos = head_pose[0]
                head_orn = Rotation.from_quat(head_pose[1])
                hand_tip_pose = right_wrist_orn.apply(right_positions) + right_wrist_pos
                hand_tip_pose = hand_tip_pose[[1,2,3,0]]
                right_arm_q, right_hand_q = quest.solve_system_world(right_wrist_pos, right_wrist_orn, hand_tip_pose)
                quest.send_ik_result(right_arm_q, right_hand_q)
                if quest.data_dir is not None:
                    message = robot_interface.control(control_type="JOINT_IMPEDANCE",
                                                    action=right_arm_q,
                                                    mode=0.0,
                                                    controller_cfg=controller_cfg)
                    redis_client.set('right_leap_action', pickle.dumps(convert_to_hardware(right_hand_q)))
        except socket.error as e:
            print(e)
            pass
        except KeyboardInterrupt:
            #camera.close()
            rokoko.close()
            quest.close()
            break
        else:
            packet_time = time.time()
            fps_counter += 1
            packet_counter += 1

            if (packet_time - start_time) > 1.0:
                print(f"received {fps_counter} packets in a second", end="\r")
                start_time += 1.0
                fps_counter = 0
        

