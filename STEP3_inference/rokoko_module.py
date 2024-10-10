import socket
import json
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as pb
from rigidbodySento import create_primitive_shape

class RokokoModule:
    hand_link_names = ["Hand",
                   "ThumbProximal","ThumbMedial","ThumbDistal","ThumbTip",
                   "IndexProximal","IndexMedial","IndexDistal","IndexTip",
                   "MiddleProximal","MiddleMedial","MiddleDistal","MiddleTip",
                   "RingProximal","RingMedial","RingDistal","RingTip",
                   "LittleProximal","LittleMedial","LittleDistal","LittleTip"]
    
    first_kps = [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
    second_kps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    scale = [1,1,0.7,0.7,1,0.7,0.7,0.7,1,0.7,0.7,0.7,1,0.7,0.7,0.7,1,0.7,0.7,0.7]

    tip_id = [[4, 8, 12, 16, 20],[3,7,11,15,19],[2,6,10,14,18],[1,5,9,13,17],[0,0,0,0,0]]
    #tip_id = list(range(21))
    def __init__(self,vr_ip, hand_info_port ,listener_port, visualization=False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)
        self.sock.bind(("", listener_port))
        self.sock.setblocking(1)
        self.tip_vis_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tip_vis_dest = (vr_ip, hand_info_port)
        self.visualization = visualization
        self.cnt = 0
        if self.visualization:
            pb.connect(pb.GUI)
            self.vis_sp_left = [create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, (0.01,), collidable=False) for _ in range(21)]
            self.vis_sp_right = [create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, (0.01,), collidable=False) for _ in range(21)]

    def round_floats(self, o):
        if isinstance(o, float):
            return round(o, 5)
        if isinstance(o, dict):
            return {k: self.round_floats(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.round_floats(x) for x in o]
        return o

    def Unpack(self, data):
        info = json.loads(data)
        left_link_positions = []
        if "leftHand" in info["scene"]["actors"][0]["body"].keys():
            raw_wrist_orn = info["scene"]["actors"][0]["body"]["leftHand"]["rotation"]
            wrist_orn = Rotation.from_quat(np.array([raw_wrist_orn["x"], raw_wrist_orn["y"], raw_wrist_orn["z"], raw_wrist_orn["w"]]))
            raw_wrist_position = info["scene"]["actors"][0]["body"]["leftHand"]["position"]
            wrist_position = np.array([raw_wrist_position["x"], raw_wrist_position["y"], raw_wrist_position["z"]])
            for link_name in RokokoModule.hand_link_names:
                link_name = "left" + link_name
                raw_pos = info["scene"]["actors"][0]["body"][link_name]["position"]
                pos = np.array([raw_pos["x"], raw_pos["y"], raw_pos["z"]])
                rel_pos = wrist_orn.inv().apply(pos - wrist_position)
                left_link_positions.append(rel_pos)
            left_link_positions = -np.array(left_link_positions)
        right_link_positions = []
        if "rightHand" in info["scene"]["actors"][0]["body"].keys():
            raw_wrist_orn = info["scene"]["actors"][0]["body"]["rightHand"]["rotation"]
            wrist_orn = Rotation.from_quat(np.array([raw_wrist_orn["x"], raw_wrist_orn["y"], raw_wrist_orn["z"], raw_wrist_orn["w"]]))
            raw_wrist_position = info["scene"]["actors"][0]["body"]["rightHand"]["position"]
            wrist_position = np.array([raw_wrist_position["x"], raw_wrist_position["y"], raw_wrist_position["z"]])
            for link_name in RokokoModule.hand_link_names:
                link_name = "right" + link_name
                raw_pos = info["scene"]["actors"][0]["body"][link_name]["position"]
                pos = np.array([raw_pos["x"], raw_pos["y"], raw_pos["z"]])
                rel_pos = wrist_orn.inv().apply(pos - wrist_position)
                right_link_positions.append(rel_pos)
            right_link_positions = -np.array(right_link_positions)
        left_link_positions, right_link_positions = self.adjust_bone_length(left_link_positions, right_link_positions, RokokoModule.scale)
        tip_id = RokokoModule.tip_id[self.cnt]
        # self.cnt += 1
        # if self.cnt == 4:
        #     self.cnt = 0
        return left_link_positions[tip_id], right_link_positions[tip_id]

    def adjust_bone_length(self, left_positions, right_positions, factors = [1.0]*20):
        factors = np.array(factors) * 1.3
        if left_positions.shape[0] == 21:
            diff = left_positions[RokokoModule.second_kps] - left_positions[RokokoModule.first_kps]
            diff = diff * factors.reshape((-1,1))
            diff = diff.reshape((5,4,3))
            new_left_positions = diff.cumsum(axis=1).reshape(-1,3)
            left_positions = np.vstack([left_positions[0], new_left_positions])
        if right_positions.shape[0] == 21:
            diff = right_positions[RokokoModule.second_kps] - right_positions[RokokoModule.first_kps]
            diff = diff * factors.reshape((-1,1))
            diff = diff.reshape((5,4,3))
            new_right_positions = diff.cumsum(axis=1).reshape(-1,3)
            right_positions = np.vstack([right_positions[0], new_right_positions])
        return left_positions, right_positions

    def send_joint_data(self, tip_positions):
        """
        tip_positions: list of lists of 3 floats
        conn: socket connector
        """
        msg = {}
        msg["lt"] = tip_positions[0].tolist()
        msg["li"] = tip_positions[1].tolist()
        msg["lm"] = tip_positions[2].tolist()
        msg["lr"] = tip_positions[3].tolist()
        msg["rt"] = tip_positions[5].tolist()
        msg["ri"] = tip_positions[6].tolist()
        msg["rm"] = tip_positions[7].tolist()
        msg["rr"] = tip_positions[8].tolist()
        json_message = json.dumps(self.round_floats(msg))
        self.tip_vis_sock.sendto(json_message.encode(), self.tip_vis_dest)

    def receive(self):
        data, _ = self.sock.recvfrom(40000)
        left_positions, right_positions = self.Unpack(data.decode())
        return left_positions, right_positions
    
    def visualize_hand(self, left_positions, right_positions):
        for i in range(len(left_positions)):
            pb.resetBasePositionAndOrientation(self.vis_sp_left[i], left_positions[i], (0,0,0,1))
        for i in range(len(right_positions)):
            pb.resetBasePositionAndOrientation(self.vis_sp_right[i], right_positions[i], (0,0,0,1))
    
    def close(self):
        self.sock.close()

if __name__ == "__main__":
    rokoko = RokokoModule("10.5.70.193",65432, 14043, visualization=True)
    scale = [1,0.7,0.7,0.7,1,0.7,0.7,0.7,1,0.7,0.7,0.7,1,0.7,0.7,0.7,1,0.7,0.7,0.7]
    while True:
        left_positions, right_positions = rokoko.receive()
        left_positions, right_positions = rokoko.adjust_bone_length(left_positions, right_positions, scale)
        rokoko.visualize_hand([], right_positions)
        #rokoko.send_joint_data(np.vstack([left_positions, right_positions]))