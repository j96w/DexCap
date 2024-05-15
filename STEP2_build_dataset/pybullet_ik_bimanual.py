import pybullet_data
from yourdfpy import URDF
from transforms3d.euler import quat2euler, euler2quat
from utils import *
from hyperparameters import *

class LeapPybulletIK():
    def __init__(self):
        # start pybullet
        clid = p.connect(p.SHARED_MEMORY)
        if clid < 0:
            p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, -0.3])

        # load right LEAP hand
        self.LeapId = p.loadURDF(
            "leap_hand_mesh/robot_pybullet.urdf",
            [0.0, 0.0, 0.0],
            rotate_quaternion(0.0, 0.0, 0.0),
        )

        # load left LEAP hand
        self.left_offset = 1.0 # for visualization, separate left and right hand
        self.LeapId_2 = p.loadURDF(
            "leap_hand_mesh/robot_pybullet.urdf",
            [0.0, self.left_offset, 0.0],
            rotate_quaternion(0.0, 0.0, 0.0),
        )

        self.leap_center_offset = [0.18, 0.03, 0.0] # Since the root of the LEAP hand URDF is not at the palm's root (it is at the root of the index finger), we set an offset to correct the root location
        self.leapEndEffectorIndex = [4, 9, 14, 19] # fingertip joint index
        self.fingertip_offset = np.array([0.1, 0.0, -0.08]) # Since the root of the fingertip mesh in URDF is not at the tip (it is at the right lower part of the fingertip mesh), we set an offset to correct the fingertip location
        self.thumb_offset = np.array([0.1, 0.0, -0.06]) # Same reason for the thumb tip

        self.numJoints = p.getNumJoints(self.LeapId)
        self.hand_lower_limits, self.hand_upper_limits, self.hand_joint_ranges = self.get_joint_limits(self.LeapId) # get the joint limits of LEAP hand
        self.HAND_Q = np.array([np.pi / 6, -np.pi / 6, np.pi / 3, np.pi / 3,
                               np.pi / 6, 0.0, np.pi / 3, np.pi / 3,
                               np.pi / 6, np.pi / 6, np.pi / 3, np.pi / 3,
                               np.pi / 6, np.pi / 6, np.pi / 3, np.pi / 3]) # To avoid self-collision of LEAP hand, we define a reference pose for null space IK

        # load URDF of left and right hand for generating pointcloud during forward kinematics
        self.urdf_dict = {}
        self.Leap_urdf = URDF.load("leap_hand_mesh/robot_pybullet.urdf")
        self.Leap_urdf_2 = URDF.load("leap_hand_mesh/robot_pybullet.urdf")
        self.urdf_dict["right_leap"] = {
            "scene": self.Leap_urdf.scene,
            "mesh_list": self._load_meshes(self.Leap_urdf.scene),
        }
        self.urdf_dict["left_leap"] = {
            "scene": self.Leap_urdf_2.scene,
            "mesh_list": self._load_meshes(self.Leap_urdf_2.scene),
        }

        self.create_target_vis()
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)

    def get_joint_limits(self, robot):
        joint_lower_limits = []
        joint_upper_limits = []
        joint_ranges = []
        for i in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, i)
            if joint_info[2] == p.JOINT_FIXED:
                continue
            joint_lower_limits.append(joint_info[8])
            joint_upper_limits.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])
        return joint_lower_limits, joint_upper_limits, joint_ranges

    def _load_meshes(self, scene):
        mesh_list = []
        for name, g in scene.geometry.items():
            mesh = g.as_open3d
            mesh_list.append(mesh)

        return mesh_list

    def _update_meshes(self, type):
        mesh_new = o3d.geometry.TriangleMesh()
        for idx, name in enumerate(self.urdf_dict[type]["scene"].geometry.keys()):
            mesh_new += copy.deepcopy(self.urdf_dict[type]["mesh_list"][idx]).transform(
                self.urdf_dict[type]["scene"].graph.get(name)[0]
            )
        return mesh_new

    def get_mesh_pointcloud(self, joint_pos, joint_pos_left):
        self.Leap_urdf.update_cfg(joint_pos)
        right_mesh = self._update_meshes("right_leap")  # Get the new updated mesh
        robot_pc = right_mesh.sample_points_uniformly(number_of_points=80000)

        self.Leap_urdf_2.update_cfg(joint_pos_left)
        left_mesh = self._update_meshes("left_leap")  # Get the new updated mesh
        robot_pc_left = left_mesh.sample_points_uniformly(number_of_points=80000)

        # Convert the sampled mesh point cloud to the format expected by Open3D
        new_points = np.asarray(robot_pc.points)  # Convert to numpy array for points
        new_points_left = np.asarray(robot_pc_left.points)  # Convert to numpy array for points
        new_points_left[:, 1] = -1.0 * new_points_left[:, 1] # flip the right hand mesh to left hand mesh

        return new_points, new_points_left

    def switch_vector_from_rokoko(self, vector):
        return [vector[0], -vector[2], vector[1]]

    def post_process_rokoko_pos(self, rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos):
        rightHandThumb_pos[-1] *= -1.0
        rightHandThumb_pos = self.switch_vector_from_rokoko(rightHandThumb_pos)
        rightHandIndex_pos[-1] *= -1.0
        rightHandIndex_pos = self.switch_vector_from_rokoko(rightHandIndex_pos)
        rightHandMiddle_pos[-1] *= -1.0
        rightHandMiddle_pos = self.switch_vector_from_rokoko(rightHandMiddle_pos)
        rightHandRing_pos[-1] *= -1.0
        rightHandRing_pos = self.switch_vector_from_rokoko(rightHandRing_pos)

        return rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos

    def post_process_rokoko_ori(self, input_quat):
        wxyz_input_quat = np.array([input_quat[3], input_quat[0], input_quat[1], input_quat[2]])
        wxyz_input_mat = quat2mat(wxyz_input_quat)

        rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        wxyz_input_mat = np.dot(wxyz_input_mat, rot_mat)

        wxyz_transform_quat = mat2quat(wxyz_input_mat)
        xyzw_transform_quat = np.array([wxyz_transform_quat[1], wxyz_transform_quat[2], wxyz_transform_quat[3], wxyz_transform_quat[0]])

        return xyzw_transform_quat

    def create_target_vis(self):

        # load balls (used for visualization)
        small_ball_radius = 0.001
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius) # small ball used to indicate fingertip current position
        ball_radius = 0.02
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius) # large ball used to indicate fingertip goal position
        baseMass = 0.001
        basePosition = [0, 0, 0]

        self.ball1Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition) # for base and finger tip joints
        self.ball2Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
        self.ball3Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
        self.ball4Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
        self.ball5Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball6Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball7Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball8Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball9Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball10Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)

        self.ball11Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition) # for base and finger tip joints
        self.ball12Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
        self.ball13Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
        self.ball14Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
        self.ball15Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball16Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball17Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball18Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball19Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
        self.ball20Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)

        p.changeVisualShape(self.ball1Mbt, -1, rgbaColor=[1, 0, 0, 1])  # Red
        p.changeVisualShape(self.ball2Mbt, -1, rgbaColor=[0, 1, 0, 1])  # Green
        p.changeVisualShape(self.ball3Mbt, -1, rgbaColor=[0, 0, 1, 1])  # Blue
        p.changeVisualShape(self.ball4Mbt, -1, rgbaColor=[1, 1, 0, 1])  # Yellow
        p.changeVisualShape(self.ball5Mbt, -1, rgbaColor=[1, 1, 1, 1])  # White
        p.changeVisualShape(self.ball6Mbt, -1, rgbaColor=[0, 0, 0, 1])  # Black
        p.changeVisualShape(self.ball7Mbt, -1, rgbaColor=[1, 0, 0, 1])  # Red
        p.changeVisualShape(self.ball8Mbt, -1, rgbaColor=[0, 1, 0, 1])  # Green
        p.changeVisualShape(self.ball9Mbt, -1, rgbaColor=[0, 0, 1, 1])  # Blue
        p.changeVisualShape(self.ball10Mbt, -1, rgbaColor=[1, 1, 0, 1])  # Yellow

        p.changeVisualShape(self.ball11Mbt, -1, rgbaColor=[1, 0, 0, 1])  # Red
        p.changeVisualShape(self.ball12Mbt, -1, rgbaColor=[0, 1, 0, 1])  # Green
        p.changeVisualShape(self.ball13Mbt, -1, rgbaColor=[0, 0, 1, 1])  # Blue
        p.changeVisualShape(self.ball14Mbt, -1, rgbaColor=[1, 1, 0, 1])  # Yellow
        p.changeVisualShape(self.ball15Mbt, -1, rgbaColor=[1, 1, 1, 1])  # White
        p.changeVisualShape(self.ball16Mbt, -1, rgbaColor=[0, 0, 0, 1])  # Black
        p.changeVisualShape(self.ball17Mbt, -1, rgbaColor=[1, 0, 0, 1])  # Red
        p.changeVisualShape(self.ball18Mbt, -1, rgbaColor=[0, 1, 0, 1])  # Green
        p.changeVisualShape(self.ball19Mbt, -1, rgbaColor=[0, 0, 1, 1])  # Blue
        p.changeVisualShape(self.ball20Mbt, -1, rgbaColor=[1, 1, 0, 1])  # Yellow

        no_collision_group = 0
        no_collision_mask = 0
        p.setCollisionFilterGroupMask(self.ball1Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball2Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball3Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball4Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball5Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball6Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball7Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball8Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball9Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball10Mbt, -1, no_collision_group, no_collision_mask)

        p.setCollisionFilterGroupMask(self.ball11Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball12Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball13Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball14Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball15Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball16Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball17Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball18Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball19Mbt, -1, no_collision_group, no_collision_mask)
        p.setCollisionFilterGroupMask(self.ball20Mbt, -1, no_collision_group, no_collision_mask)

    def update_target_vis(self, rightHand_rot, rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos):
        p.resetBasePositionAndOrientation(
            self.ball6Mbt,
            rotate_vector_by_quaternion_using_matrix(self.leap_center_offset, rightHand_rot),
            rightHand_rot,
        )
        p.resetBasePositionAndOrientation(
            self.ball5Mbt,
            [0.0, 0.0, 0.0],
            rightHand_rot,
        )

        p.resetBasePositionAndOrientation(
            self.ball9Mbt,
            p.getLinkState(self.LeapId, 4)[0],
            rightHand_rot,
        )
        p.resetBasePositionAndOrientation(
            self.ball7Mbt,
            p.getLinkState(self.LeapId, 9)[0],
            rightHand_rot,
        )
        p.resetBasePositionAndOrientation(
            self.ball8Mbt,
            p.getLinkState(self.LeapId, 14)[0],
            rightHand_rot,
        )
        p.resetBasePositionAndOrientation(
            self.ball10Mbt,
            p.getLinkState(self.LeapId, 19)[0],
            rightHand_rot,
        )

        offset = rotate_vector_by_quaternion_using_matrix(self.fingertip_offset, rightHand_rot)
        thumb_offset = rotate_vector_by_quaternion_using_matrix(self.thumb_offset, rightHand_rot)

        rightHandThumb_pos += thumb_offset
        _, current_orientation = p.getBasePositionAndOrientation(self.ball4Mbt)
        p.resetBasePositionAndOrientation(
            self.ball4Mbt, rightHandThumb_pos, current_orientation
        )
        rightHandIndex_pos += offset
        _, current_orientation = p.getBasePositionAndOrientation(self.ball3Mbt)
        p.resetBasePositionAndOrientation(
            self.ball3Mbt, rightHandIndex_pos, current_orientation
        )
        rightHandMiddle_pos += offset
        _, current_orientation = p.getBasePositionAndOrientation(self.ball1Mbt)
        p.resetBasePositionAndOrientation(
            self.ball1Mbt, rightHandMiddle_pos, current_orientation
        )
        rightHandRing_pos += offset
        _, current_orientation = p.getBasePositionAndOrientation(self.ball2Mbt)
        p.resetBasePositionAndOrientation(
            self.ball2Mbt, rightHandRing_pos, current_orientation
        )

        return rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos

    def update_target_vis_left(self, leftHand_rot, leftHandThumb_pos, leftHandIndex_pos, leftHandMiddle_pos, leftHandRing_pos):

        after_left_offset = rotate_vector_by_quaternion_using_matrix(self.leap_center_offset, leftHand_rot)
        after_left_offset[1] += self.left_offset
        p.resetBasePositionAndOrientation(
            self.ball16Mbt,
            after_left_offset,
            leftHand_rot,
        )
        p.resetBasePositionAndOrientation(
            self.ball15Mbt,
            [0.0, self.left_offset, 0.0],
            leftHand_rot,
        )

        p.resetBasePositionAndOrientation(
            self.ball19Mbt,
            p.getLinkState(self.LeapId_2, 4)[0],
            leftHand_rot,
        )
        p.resetBasePositionAndOrientation(
            self.ball17Mbt,
            p.getLinkState(self.LeapId_2, 9)[0],
            leftHand_rot,
        )
        p.resetBasePositionAndOrientation(
            self.ball18Mbt,
            p.getLinkState(self.LeapId_2, 14)[0],
            leftHand_rot,
        )
        p.resetBasePositionAndOrientation(
            self.ball20Mbt,
            p.getLinkState(self.LeapId_2, 19)[0],
            leftHand_rot,
        )

        leftHandThumb_pos[1] += self.left_offset
        leftHandIndex_pos[1] += self.left_offset
        leftHandMiddle_pos[1] += self.left_offset
        leftHandRing_pos[1] += self.left_offset

        offset = rotate_vector_by_quaternion_using_matrix(self.fingertip_offset, leftHand_rot)
        thumb_offset = rotate_vector_by_quaternion_using_matrix(self.thumb_offset, leftHand_rot)

        leftHandThumb_pos += thumb_offset
        _, current_orientation = p.getBasePositionAndOrientation(self.ball14Mbt)
        p.resetBasePositionAndOrientation(
            self.ball14Mbt, leftHandThumb_pos, current_orientation
        )
        leftHandIndex_pos += offset
        _, current_orientation = p.getBasePositionAndOrientation(self.ball3Mbt)
        p.resetBasePositionAndOrientation(
            self.ball13Mbt, leftHandIndex_pos, current_orientation
        )
        leftHandMiddle_pos += offset
        _, current_orientation = p.getBasePositionAndOrientation(self.ball1Mbt)
        p.resetBasePositionAndOrientation(
            self.ball11Mbt, leftHandMiddle_pos, current_orientation
        )
        leftHandRing_pos += offset
        _, current_orientation = p.getBasePositionAndOrientation(self.ball2Mbt)
        p.resetBasePositionAndOrientation(
            self.ball12Mbt, leftHandRing_pos, current_orientation
        )

        return leftHandThumb_pos, leftHandIndex_pos, leftHandMiddle_pos, leftHandRing_pos

    def rest_target_vis(self):
        p.resetBaseVelocity(self.ball1Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball2Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball3Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball4Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball5Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball6Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball7Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball8Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball9Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball10Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball11Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball12Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball13Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball14Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball15Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball16Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball17Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball18Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball19Mbt, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.ball20Mbt, [0, 0, 0], [0, 0, 0])

    def compute_IK(self, right_hand_pos, right_hand_wrist_ori, left_hand_pos, left_hand_wrist_ori):
        p.stepSimulation()

        wxyz_input_quat = np.array([left_hand_wrist_ori[3], left_hand_wrist_ori[0], left_hand_wrist_ori[1], left_hand_wrist_ori[2]])
        wxyz_input_mat = quat2mat(wxyz_input_quat)

        leftHand_pos = left_hand_pos[0]
        leftHandThumb_pos = (left_hand_pos[4] - leftHand_pos)
        leftHandIndex_pos = (left_hand_pos[8] - leftHand_pos)
        leftHandMiddle_pos = (left_hand_pos[12] - leftHand_pos)
        leftHandRing_pos = (left_hand_pos[16] - leftHand_pos)

        leftHandThumb_pos = leftHandThumb_pos @ wxyz_input_mat
        leftHandIndex_pos = leftHandIndex_pos @ wxyz_input_mat
        leftHandMiddle_pos = leftHandMiddle_pos @ wxyz_input_mat
        leftHandRing_pos = leftHandRing_pos @ wxyz_input_mat

        leftHandThumb_pos[0] *= -1.0
        leftHandIndex_pos[0] *= -1.0
        leftHandMiddle_pos[0] *= -1.0
        leftHandRing_pos[0] *= -1.0

        leftHandThumb_pos = leftHandThumb_pos @ wxyz_input_mat.T
        leftHandIndex_pos = leftHandIndex_pos @ wxyz_input_mat.T
        leftHandMiddle_pos = leftHandMiddle_pos @ wxyz_input_mat.T
        leftHandRing_pos = leftHandRing_pos @ wxyz_input_mat.T

        # transform left hand orientation
        leftHand_rot = left_hand_wrist_ori
        leftHand_rot = self.post_process_rokoko_ori(leftHand_rot)
        euler_angles = quat2euler(np.array([leftHand_rot[3], leftHand_rot[0], leftHand_rot[1], leftHand_rot[2]]))
        quat_angles = euler2quat(-euler_angles[0], -euler_angles[1], euler_angles[2]).tolist()
        leftHand_rot = np.array(quat_angles[1:] + quat_angles[:1])
        leftHand_rot = rotate_quaternion_xyzw(leftHand_rot, np.array([1.0, 0.0, 0.0]), np.pi / 2.0)

        # get right hand position information including fingers
        rightHand_pos = right_hand_pos[0]
        rightHandThumb_pos = (right_hand_pos[4] - rightHand_pos)
        rightHandIndex_pos = (right_hand_pos[8] - rightHand_pos)
        rightHandMiddle_pos = (right_hand_pos[12] - rightHand_pos)
        rightHandRing_pos = (right_hand_pos[16] - rightHand_pos)

        leftHandThumb_pos, leftHandIndex_pos, leftHandMiddle_pos, leftHandRing_pos = self.post_process_rokoko_pos(leftHandThumb_pos, leftHandIndex_pos, leftHandMiddle_pos, leftHandRing_pos)
        leftHandThumb_pos, leftHandIndex_pos, leftHandMiddle_pos, leftHandRing_pos = self.update_target_vis_left(leftHand_rot, leftHandThumb_pos, leftHandIndex_pos, leftHandMiddle_pos, leftHandRing_pos)

        leapEndEffectorPos_2 = [
            leftHandIndex_pos,
            leftHandMiddle_pos,
            leftHandRing_pos,
            leftHandThumb_pos
        ]

        # transform right hand orientation
        rightHand_rot = right_hand_wrist_ori
        rightHand_rot = self.post_process_rokoko_ori(rightHand_rot)
        euler_angles = quat2euler(np.array([rightHand_rot[3], rightHand_rot[0], rightHand_rot[1], rightHand_rot[2]]))
        quat_angles = euler2quat(-euler_angles[0], -euler_angles[1], euler_angles[2]).tolist()
        rightHand_rot = np.array(quat_angles[1:] + quat_angles[:1])
        rightHand_rot = rotate_quaternion_xyzw(rightHand_rot, np.array([1.0, 0.0, 0.0]), np.pi / 2.0)

        rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos = self.post_process_rokoko_pos(rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos)
        rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos = self.update_target_vis(rightHand_rot, rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos)

        leapEndEffectorPos = [
            rightHandIndex_pos,
            rightHandMiddle_pos,
            rightHandRing_pos,
            rightHandThumb_pos
        ]

        jointPoses_2 = []
        for i in range(4):
            jointPoses_2 = jointPoses_2 + list(
                p.calculateInverseKinematics(self.LeapId_2, self.leapEndEffectorIndex[i], leapEndEffectorPos_2[i],
                                      lowerLimits=self.hand_lower_limits, upperLimits=self.hand_upper_limits, jointRanges=self.hand_joint_ranges,
                                      restPoses=self.HAND_Q.tolist(), maxNumIterations=1000, residualThreshold=0.001))[4 * i:4 * (i + 1)]
        jointPoses_2 = tuple(jointPoses_2)

        jointPoses = []
        for i in range(4):
            jointPoses = jointPoses + list(
                p.calculateInverseKinematics(self.LeapId, self.leapEndEffectorIndex[i], leapEndEffectorPos[i],
                                      lowerLimits=self.hand_lower_limits, upperLimits=self.hand_upper_limits, jointRanges=self.hand_joint_ranges,
                                      restPoses=self.HAND_Q.tolist(), maxNumIterations=1000, residualThreshold=0.001))[4 * i:4 * (i + 1)]
        jointPoses = tuple(jointPoses)

        combined_jointPoses_2 = (jointPoses_2[0:4] + (0.0,) + jointPoses_2[4:8] + (0.0,) + jointPoses_2[8:12] + (0.0,) + jointPoses_2[12:16] + (0.0,))
        combined_jointPoses_2 = list(combined_jointPoses_2)
        combined_jointPoses = (jointPoses[0:4] + (0.0,) + jointPoses[4:8] + (0.0,) + jointPoses[8:12] + (0.0,) + jointPoses[12:16] + (0.0,))
        combined_jointPoses = list(combined_jointPoses)

        # update the hand joints
        for i in range(20):
            p.setJointMotorControl2(
                bodyIndex=self.LeapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

            p.setJointMotorControl2(
                bodyIndex=self.LeapId_2,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses_2[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        p.resetBasePositionAndOrientation(
            self.LeapId,
            rotate_vector_by_quaternion_using_matrix(self.leap_center_offset, rightHand_rot),
            rightHand_rot,
        )

        after_left_offset_base = rotate_vector_by_quaternion_using_matrix(self.leap_center_offset, leftHand_rot)
        after_left_offset_base[1] += self.left_offset
        p.resetBasePositionAndOrientation(
            self.LeapId_2,
            after_left_offset_base,
            leftHand_rot,
        )

        self.rest_target_vis()

        # map results to real robot
        real_right_robot_hand_q = np.array([0.0 for _ in range(16)])
        real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

        real_right_robot_hand_q[0:4] = jointPoses[0:4]
        real_right_robot_hand_q[4:8] = jointPoses[4:8]
        real_right_robot_hand_q[8:12] = jointPoses[8:12]
        real_right_robot_hand_q[12:16] = jointPoses[12:16]
        real_right_robot_hand_q[0:2] = real_right_robot_hand_q[0:2][::-1]
        real_right_robot_hand_q[4:6] = real_right_robot_hand_q[4:6][::-1]
        real_right_robot_hand_q[8:10] = real_right_robot_hand_q[8:10][::-1]

        real_left_robot_hand_q[0:4] = jointPoses_2[0:4]
        real_left_robot_hand_q[4:8] = jointPoses_2[4:8]
        real_left_robot_hand_q[8:12] = jointPoses_2[8:12]
        real_left_robot_hand_q[12:16] = jointPoses_2[12:16]
        real_left_robot_hand_q[0:2] = real_left_robot_hand_q[0:2][::-1]
        real_left_robot_hand_q[4:6] = real_left_robot_hand_q[4:6][::-1]
        real_left_robot_hand_q[8:10] = real_left_robot_hand_q[8:10][::-1]

        # generate pointcloud of the left and right hand with forward kinematics
        right_hand_pointcloud, left_hand_pointcloud = self.get_mesh_pointcloud(real_right_robot_hand_q, real_left_robot_hand_q)

        # further map joints to real robot
        real_right_robot_hand_q += np.pi
        real_left_robot_hand_q += np.pi
        real_left_robot_hand_q[0] = np.pi * 2 - real_left_robot_hand_q[0]
        real_left_robot_hand_q[4] = np.pi * 2 - real_left_robot_hand_q[4]
        real_left_robot_hand_q[8] = np.pi * 2 - real_left_robot_hand_q[8]
        real_left_robot_hand_q[12] = np.pi * 2 - real_left_robot_hand_q[12]
        real_left_robot_hand_q[13] = np.pi * 2 - real_left_robot_hand_q[13]

        return real_right_robot_hand_q, real_left_robot_hand_q, right_hand_pointcloud, left_hand_pointcloud