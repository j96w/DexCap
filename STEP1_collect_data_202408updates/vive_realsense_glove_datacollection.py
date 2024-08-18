import os
import ctypes
from ctypes import cast, byref
import time
import xr
import numpy as np
import pyrealsense2 as rs
from open3d_vis_obj import VIVEOpen3DVisualizer
from openxr_utils import ContextObject
import recording_utils as utils
import cv2
import redis


def main(dataset_folder):
    dataset_folder, data_folder = utils.setup_folder_structure(dataset_folder)

    # Initialize Redis connection
    redis_host = "localhost"
    redis_port = 6669
    redis_password = ""  # If your Redis server has no password, keep it as an empty string.
    r = redis.StrictRedis(
        host=redis_host, port=redis_port, password=redis_password, decode_responses=False
    )

    visualizer = VIVEOpen3DVisualizer()
    first = True
    first_2 = True
    first_3 = True

    pipeline, pipeline_profile = utils.configure_realsense()

    intrinsics = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    utils.save_camera_intrinsics(dataset_folder, intrinsics)

    align = rs.align(rs.stream.color)

    with ContextObject(
        instance_create_info=xr.InstanceCreateInfo(
            enabled_extension_names=[
                xr.MND_HEADLESS_EXTENSION_NAME,
                xr.extension.HTCX_vive_tracker_interaction.NAME,
            ],
        ),
    ) as context:
        instance = context.instance
        session = context.session

        enumerateViveTrackerPathsHTCX = cast(
            xr.get_instance_proc_addr(instance, "xrEnumerateViveTrackerPathsHTCX"),
            xr.PFN_xrEnumerateViveTrackerPathsHTCX
        )

        role_strings = [
            "handheld_object", "left_foot", "right_foot", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_knee", "right_knee", "waist", "chest",
            "camera", "keyboard",
        ]
        role_path_strings = [f"/user/vive_tracker_htcx/role/{role}" for role in role_strings]
        role_paths = (xr.Path * len(role_path_strings))(*[xr.string_to_path(instance, role_string) for role_string in role_path_strings])
        pose_action = xr.create_action(
            action_set=context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="tracker_pose",
                localized_action_name="Tracker Pose",
                count_subaction_paths=len(role_paths),
                subaction_paths=role_paths,
            ),
        )
        suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(
            *[xr.ActionSuggestedBinding(pose_action, xr.string_to_path(instance, f"{role_path_string}/input/grip/pose")) for role_path_string in role_path_strings]
        )
        xr.suggest_interaction_profile_bindings(instance=instance, suggested_bindings=xr.InteractionProfileSuggestedBinding(
            interaction_profile=xr.string_to_path(instance, "/interaction_profiles/htc/vive_tracker_htcx"),
            count_suggested_bindings=len(suggested_binding_paths), suggested_bindings=suggested_binding_paths,
        ))
        tracker_action_spaces = (xr.Space * len(role_paths))(
            *[xr.create_action_space(session=session, create_info=xr.ActionSpaceCreateInfo(action=pose_action, subaction_path=role_path)) for role_path in role_paths]
        )

        n_paths = ctypes.c_uint32(0)
        result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
        if xr.check_result(result).is_exception():
            raise result
        vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
        result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
        if xr.check_result(result).is_exception():
            raise result
        print(xr.Result(result), n_paths.value)

        session_was_focused = False
        frame_index = 0

        for frame_state in context.frame_loop():
            if context.session_state == xr.SessionState.FOCUSED:
                session_was_focused = True
                active_action_set = xr.ActiveActionSet(action_set=context.default_action_set, subaction_path=xr.NULL_PATH)
                xr.sync_actions(session=session, sync_info=xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=ctypes.pointer(active_action_set)))

                n_paths = ctypes.c_uint32(0)
                result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
                if xr.check_result(result).is_exception():
                    raise result
                vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
                result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
                if xr.check_result(result).is_exception():
                    raise result
                found_tracker_count = 0

                for index, space in enumerate(tracker_action_spaces):
                    space_location = xr.locate_space(space=space, base_space=context.space, time=frame_state.predicted_display_time)
                    if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                        if role_strings[index] == 'right_elbow':
                            if first:
                                visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 0)
                                first = False
                            else:
                                visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 0)
                        elif role_strings[index] == 'left_elbow':
                            if first_2:
                                visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 1)
                                first_2 = False
                            else:
                                visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 1)
                        elif role_strings[index] == 'chest':
                            if first_3:
                                visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 2)
                                first_3 = False
                            else:
                                visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 2)

                        frame_folder = os.path.join(data_folder, f"frame_{frame_index:04d}")
                        os.makedirs(frame_folder, exist_ok=True)
                        if role_strings[index] == 'right_elbow':
                            utils.save_pose(os.path.join(frame_folder, "right_pose.txt"), [space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z])
                        elif role_strings[index] == 'left_elbow':
                            utils.save_pose(os.path.join(frame_folder, "left_pose.txt"), [space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z])
                        elif role_strings[index] == 'chest':
                            utils.save_pose(os.path.join(frame_folder, "chest_pose.txt"), [space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z])

                        found_tracker_count += 1

                if found_tracker_count == 0:
                    print("no trackers found")
                    
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                utils.save_image(os.path.join(frame_folder, "color.png"), color_image)
                cv2.imwrite(os.path.join(frame_folder, "depth.png"), depth_image)

                # Retrieve and save hand joint data from Redis
                raw_left_hand_joint_xyz = np.frombuffer(r.get("rawLeftHandJointXyz"), dtype=np.float64).reshape((21, 3))
                raw_right_hand_joint_xyz = np.frombuffer(r.get("rawRightHandJointXyz"), dtype=np.float64).reshape((21, 3))
                raw_left_hand_joint_orientation = np.frombuffer(r.get("rawLeftHandJointOrientation"), dtype=np.float64).reshape((21, 4))
                raw_right_hand_joint_orientation = np.frombuffer(r.get("rawRightHandJointOrientation"), dtype=np.float64).reshape((21, 4))

                np.savetxt(os.path.join(frame_folder, "raw_left_hand_joint_xyz.txt"), raw_left_hand_joint_xyz)
                np.savetxt(os.path.join(frame_folder, "raw_right_hand_joint_xyz.txt"), raw_right_hand_joint_xyz)
                np.savetxt(os.path.join(frame_folder, "raw_left_hand_joint_orientation.txt"), raw_left_hand_joint_orientation)
                np.savetxt(os.path.join(frame_folder, "raw_right_hand_joint_orientation.txt"), raw_right_hand_joint_orientation)

                frame_index += 1

    if not session_was_focused:
        print("This OpenXR session never entered the FOCUSED state. Did you wear the headset?")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record Vive Tracker and RealSense Data")
    parser.add_argument("dataset_folder", type=str, help="Folder to save the dataset")
    args = parser.parse_args()
    main(args.dataset_folder)
