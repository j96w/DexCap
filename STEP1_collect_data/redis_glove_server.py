import socket
import json
import redis
import numpy as np

# Initialize Redis connection
redis_host = "localhost"
redis_port = 6669
redis_password = ""  # If your Redis server has no password, keep it as an empty string.
r = redis.StrictRedis(
    host=redis_host, port=redis_port, password=redis_password, decode_responses=True
)

# Define the hand joint names for left and right hands
left_hand_joint_names = ["leftHand",
                         'leftThumbProximal', 'leftThumbMedial', 'leftThumbDistal', 'leftThumbTip',
                         'leftIndexProximal', 'leftIndexMedial', 'leftIndexDistal', 'leftIndexTip',
                         'leftMiddleProximal', 'leftMiddleMedial', 'leftMiddleDistal', 'leftMiddleTip',
                         'leftRingProximal', 'leftRingMedial', 'leftRingDistal', 'leftRingTip',
                         'leftLittleProximal', 'leftLittleMedial', 'leftLittleDistal', 'leftLittleTip']

right_hand_joint_names = ["rightHand",
                          'rightThumbProximal', 'rightThumbMedial', 'rightThumbDistal', 'rightThumbTip',
                          'rightIndexProximal', 'rightIndexMedial', 'rightIndexDistal', 'rightIndexTip',
                          'rightMiddleProximal', 'rightMiddleMedial', 'rightMiddleDistal', 'rightMiddleTip',
                          'rightRingProximal', 'rightRingMedial', 'rightRingDistal', 'rightRingTip',
                          'rightLittleProximal', 'rightLittleMedial', 'rightLittleDistal', 'rightLittleTip']

def normalize_wrt_middle_proximal(hand_positions, is_left=True):
    middle_proximal_idx = left_hand_joint_names.index('leftMiddleProximal')
    if not is_left:
        middle_proximal_idx = right_hand_joint_names.index('rightMiddleProximal')

    wrist_position = hand_positions[0]
    middle_proximal_position = hand_positions[middle_proximal_idx]
    bone_length = np.linalg.norm(wrist_position - middle_proximal_position)
    normalized_hand_positions = (middle_proximal_position - hand_positions) / bone_length
    return normalized_hand_positions


def start_server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Using SOCK_DGRAM for UDP
    s.bind(("192.168.0.200", port))
    print(f"Server started, listening on port {port} for UDP packets...")

    while True:
        data, address = s.recvfrom(64800)  # Receive UDP packets
        decoded_data = data.decode()

        # Attempt to parse JSON
        try:
            received_json = json.loads(decoded_data)

            # Initialize arrays to store the positions
            left_hand_positions = np.zeros((21, 3))
            right_hand_positions = np.zeros((21, 3))

            left_hand_orientations = np.zeros((21, 4))
            right_hand_orientations = np.zeros((21, 4))

            # Iterate through the JSON data to extract hand joint positions
            for joint_name in left_hand_joint_names:
                joint_data = received_json["scene"]["actors"][0]["body"][joint_name]
                joint_position = np.array(list(joint_data["position"].values()))
                joint_rotation = np.array(list(joint_data["rotation"].values()))
                left_hand_positions[left_hand_joint_names.index(joint_name)] = joint_position
                left_hand_orientations[left_hand_joint_names.index(joint_name)] = joint_rotation

            for joint_name in right_hand_joint_names:
                joint_data = received_json["scene"]["actors"][0]["body"][joint_name]
                joint_position = np.array(list(joint_data["position"].values()))
                joint_rotation = np.array(list(joint_data["rotation"].values()))
                right_hand_positions[right_hand_joint_names.index(joint_name)] = joint_position
                right_hand_orientations[right_hand_joint_names.index(joint_name)] = joint_rotation


            # relative distance to middle proximal joint
            # normalize by bone distance (distance from wrist to middle proximal)
            # Define the indices of 'middleProximal' in your joint names
            left_middle_proximal_idx = left_hand_joint_names.index('leftMiddleProximal')
            right_middle_proximal_idx = right_hand_joint_names.index('rightMiddleProximal')

            # Calculate bone length from 'wrist' to 'middleProximal' for both hands
            left_wrist_position = left_hand_positions[0]
            right_wrist_position = right_hand_positions[0]

            left_middle_proximal_position = left_hand_positions[left_middle_proximal_idx]
            right_middle_proximal_position = right_hand_positions[right_middle_proximal_idx]

            left_bone_length = np.linalg.norm(left_wrist_position - left_middle_proximal_position)
            right_bone_length = np.linalg.norm(right_wrist_position - right_middle_proximal_position)

            # Calculate relative positions and normalize
            normalized_left_hand_positions = (left_middle_proximal_position - left_hand_positions) / left_bone_length
            normalized_right_hand_positions = (right_middle_proximal_position - right_hand_positions) / right_bone_length

            r.set("leftHandJointXyz", np.array(normalized_left_hand_positions).astype(np.float64).tobytes())
            r.set("rightHandJointXyz", np.array(normalized_right_hand_positions).astype(np.float64).tobytes())
            r.set("rawLeftHandJointXyz", np.array(left_hand_positions).astype(np.float64).tobytes())
            r.set("rawRightHandJointXyz", np.array(right_hand_positions).astype(np.float64).tobytes())
            r.set("rawLeftHandJointOrientation", np.array(left_hand_orientations).astype(np.float64).tobytes())
            r.set("rawRightHandJointOrientation", np.array(right_hand_orientations).astype(np.float64).tobytes())


            print("\n\n")
            print("=" * 50)
            print(np.round(left_hand_positions, 3))
            print("-"*50)
            print(np.round(right_hand_positions, 3))

        except json.JSONDecodeError:
            print("Invalid JSON received:")


if __name__ == "__main__":
    start_server(14551)