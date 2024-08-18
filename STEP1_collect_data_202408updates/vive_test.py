"""
pyopenxr example program vive_tracker.py

Prints the position and orientation of your vive trackers each frame.

Helpful instructions for getting trackers working on Linux are at
https://gist.github.com/DanielArnett/c9a56c9c7cc0def20648480bca1f6772
The udev symbolic link trick was crucial in my case.
"""

import ctypes
from ctypes import cast, byref, POINTER
import time
import xr
import numpy as np
from open3d_vis_obj import VIVEOpen3DVisualizer

print("Warning: trackers with role 'Handheld object' won't be detected.")


class ContextObject(object):
    def __init__(
            self,
            instance_create_info: xr.InstanceCreateInfo = xr.InstanceCreateInfo(),
            session_create_info: xr.SessionCreateInfo = xr.SessionCreateInfo(),
            reference_space_create_info: xr.ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo(),
            view_configuration_type: xr.ViewConfigurationType = xr.ViewConfigurationType.PRIMARY_STEREO,
            environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
    ):
        self._instance_create_info = instance_create_info
        self.instance = None
        self._session_create_info = session_create_info
        self.session = None
        self.session_state = xr.SessionState.IDLE
        self._reference_space_create_info = reference_space_create_info
        self.view_configuration_type = view_configuration_type
        self.environment_blend_mode = environment_blend_mode
        self.form_factor = form_factor
        self.graphics = None
        self.graphics_binding_pointer = None
        self.action_sets = []
        self.render_layers = []
        self.swapchains = []
        self.swapchain_image_ptr_buffers = []
        self.swapchain_image_buffers = []  # Keep alive
        self.exit_render_loop = False
        self.request_restart = False  # TODO: do like hello_xr
        self.session_is_running = False

    def __enter__(self):
        self.instance = xr.create_instance(
            create_info=self._instance_create_info,
        )
        self.system_id = xr.get_system(
            instance=self.instance,
            get_info=xr.SystemGetInfo(
                form_factor=self.form_factor,
            ),
        )

        if self._session_create_info.next is not None:
            self.graphics_binding_pointer = self._session_create_info.next

        self._session_create_info.system_id = self.system_id
        self.session = xr.create_session(
            instance=self.instance,
            create_info=self._session_create_info,
        )
        self.space = xr.create_reference_space(
            session=self.session,
            create_info=self._reference_space_create_info
        )
        self.default_action_set = xr.create_action_set(
            instance=self.instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="default_action_set",
                localized_action_set_name="Default Action Set",
                priority=0,
            ),
        )
        self.action_sets.append(self.default_action_set)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.default_action_set is not None:
            xr.destroy_action_set(self.default_action_set)
            self.default_action_set = None
        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.graphics is not None:
            self.graphics.destroy()
            self.graphics = None
        if self.instance is not None:
            xr.destroy_instance(self.instance)
            self.instance = None

    def frame_loop(self):
        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=len(self.action_sets),
                action_sets=(xr.ActionSet * len(self.action_sets))(
                    *self.action_sets
                )
            ),
        )
        while True:
            self.exit_render_loop = False
            self.poll_xr_events()
            if self.exit_render_loop:
                break
            if self.session_is_running:
                if self.session_state in (
                        xr.SessionState.READY,
                        xr.SessionState.SYNCHRONIZED,
                        xr.SessionState.VISIBLE,
                        xr.SessionState.FOCUSED,
                ):
                    frame_state = xr.wait_frame(self.session)
                    xr.begin_frame(self.session)
                    self.render_layers = []

                    yield frame_state

                    xr.end_frame(
                        self.session,
                        frame_end_info=xr.FrameEndInfo(
                            display_time=frame_state.predicted_display_time,
                            environment_blend_mode=self.environment_blend_mode,
                            layers=self.render_layers,
                        )
                    )
            else:
                # Throttle loop since xrWaitFrame won't be called.
                time.sleep(0.250)

    def poll_xr_events(self):
        self.exit_render_loop = False
        self.request_restart = False
        while True:
            try:
                event_buffer = xr.poll_event(self.instance)
                event_type = xr.StructureType(event_buffer.type)
                if event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    # still handle rest of the events instead of immediately quitting
                    self.exit_render_loop = True
                    self.request_restart = True
                elif event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED \
                        and self.session is not None:
                    event = cast(
                        byref(event_buffer),
                        POINTER(xr.EventDataSessionStateChanged)).contents
                    self.session_state = xr.SessionState(event.state)
                    if self.session_state == xr.SessionState.READY:
                        xr.begin_session(
                            session=self.session,
                            begin_info=xr.SessionBeginInfo(
                                self.view_configuration_type,
                            ),
                        )
                        self.session_is_running = True
                    elif self.session_state == xr.SessionState.STOPPING:
                        self.session_is_running = False
                        xr.end_session(self.session)
                    elif self.session_state == xr.SessionState.EXITING:
                        self.exit_render_loop = True
                        self.request_restart = False
                    elif self.session_state == xr.SessionState.LOSS_PENDING:
                        self.exit_render_loop = True
                        self.request_restart = True
                elif event_type == xr.StructureType.EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX:
                    vive_tracker_connected = cast(byref(event_buffer), POINTER(xr.EventDataViveTrackerConnectedHTCX)).contents
                    paths = vive_tracker_connected.paths.contents
                    persistent_path_str = xr.path_to_string(self.instance, paths.persistent_path)
                    # print(f"Vive Tracker connected: {persistent_path_str}")
                    if paths.role_path != xr.NULL_PATH:
                        role_path_str = xr.path_to_string(self.instance, paths.role_path)
                        # print(f" New role is: {role_path_str}")
                    else:
                        # print(f" No role path.")
                        pass
                elif event_type == xr.StructureType.EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                    # print("data interaction profile changed")
                    # TODO:
                    pass
            except xr.EventUnavailable:
                break

    def view_loop(self, frame_state):
        if frame_state.should_render:
            layer = xr.CompositionLayerProjection(space=self.space)
            view_state, views = xr.locate_views(
                session=self.session,
                view_locate_info=xr.ViewLocateInfo(
                    view_configuration_type=self.view_configuration_type,
                    display_time=frame_state.predicted_display_time,
                    space=self.space,
                )
            )
            num_views = len(views)
            projection_layer_views = tuple(xr.CompositionLayerProjectionView() for _ in range(num_views))

            vsf = view_state.view_state_flags
            if (vsf & xr.VIEW_STATE_POSITION_VALID_BIT == 0
                    or vsf & xr.VIEW_STATE_ORIENTATION_VALID_BIT == 0):
                return  # There are no valid tracking poses for the views.
            for view_index, view in enumerate(views):
                view_swapchain = self.swapchains[view_index]
                swapchain_image_index = xr.acquire_swapchain_image(
                    swapchain=view_swapchain.handle,
                    acquire_info=xr.SwapchainImageAcquireInfo(),
                )
                xr.wait_swapchain_image(
                    swapchain=view_swapchain.handle,
                    wait_info=xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                )
                layer_view = projection_layer_views[view_index]
                assert layer_view.type == xr.StructureType.COMPOSITION_LAYER_PROJECTION_VIEW
                layer_view.pose = view.pose
                layer_view.fov = view.fov
                layer_view.sub_image.swapchain = view_swapchain.handle
                layer_view.sub_image.image_rect.offset[:] = [0, 0]
                layer_view.sub_image.image_rect.extent[:] = [
                    view_swapchain.width, view_swapchain.height, ]
                swapchain_image_ptr = self.swapchain_image_ptr_buffers[view_index][swapchain_image_index]
                swapchain_image = cast(swapchain_image_ptr, POINTER(xr.SwapchainImageOpenGLKHR)).contents
                assert layer_view.sub_image.image_array_index == 0  # texture arrays not supported.
                color_texture = swapchain_image.image
                self.graphics.begin_frame(layer_view, color_texture)

                yield view

                self.graphics.end_frame()
                xr.release_swapchain_image(
                    swapchain=view_swapchain.handle,
                    release_info=xr.SwapchainImageReleaseInfo()
                )
            layer.views = projection_layer_views
            self.render_layers.append(byref(layer))


# # Example usage
# translation = [1, 2, 3]
# quaternion = [1, 0, 0, 0]  # No rotation

# visualizer = Open3DVisualizer(translation, quaternion)
# visualizer.run()

visualizer = VIVEOpen3DVisualizer()
first = True
first_2 = True
first_3 = True

# ContextObject is a high level pythonic class meant to keep simple cases simple.
with ContextObject(
    instance_create_info=xr.InstanceCreateInfo(
        enabled_extension_names=[
            # A graphics extension is mandatory (without a headless extension)
            xr.MND_HEADLESS_EXTENSION_NAME,
            xr.extension.HTCX_vive_tracker_interaction.NAME,
        ],
    ),
) as context:
    instance = context.instance
    session = context.session

    # Save the function pointer
    enumerateViveTrackerPathsHTCX = cast(
        xr.get_instance_proc_addr(
            instance,
            "xrEnumerateViveTrackerPathsHTCX",
        ),
        xr.PFN_xrEnumerateViveTrackerPathsHTCX
    )

    # Create the action with subaction path
    # Role strings from
    # https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#XR_HTCX_vive_tracker_interaction
    role_strings = [
        "handheld_object",
        "left_foot",
        "right_foot",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
        "waist",
        "chest",
        "camera",
        "keyboard",
    ]
    role_path_strings = [f"/user/vive_tracker_htcx/role/{role}"
                         for role in role_strings]
    role_paths = (xr.Path * len(role_path_strings))(
        *[xr.string_to_path(instance, role_string) for role_string in role_path_strings],
    )
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
    # Describe a suggested binding for that action and subaction path
    suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(
        *[xr.ActionSuggestedBinding(
            pose_action,
            xr.string_to_path(instance, f"{role_path_string}/input/grip/pose"))
          for role_path_string in role_path_strings],
    )
    xr.suggest_interaction_profile_bindings(
        instance=instance,
        suggested_bindings=xr.InteractionProfileSuggestedBinding(
            interaction_profile=xr.string_to_path(instance, "/interaction_profiles/htc/vive_tracker_htcx"),
            count_suggested_bindings=len(suggested_binding_paths),
            suggested_bindings=suggested_binding_paths,
        )
    )
    # Create action spaces for locating trackers in each role
    tracker_action_spaces = (xr.Space * len(role_paths))(
        *[xr.create_action_space(
            session=session,
            create_info=xr.ActionSpaceCreateInfo(
                action=pose_action,
                subaction_path=role_path,
            )
        ) for role_path in role_paths],
    )

    n_paths = ctypes.c_uint32(0)
    result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
    if xr.check_result(result).is_exception():
        raise result
    vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
    # print(xr.Result(result), n_paths.value)
    result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
    if xr.check_result(result).is_exception():
        raise result
    print(xr.Result(result), n_paths.value)
    # print(*vive_tracker_paths)

    # Loop over the render frames
    session_was_focused = False  # Check for a common problem
    for frame_index, frame_state in enumerate(context.frame_loop()):
        # print(context.session_state)
        if context.session_state == xr.SessionState.FOCUSED:
            session_was_focused = True
            active_action_set = xr.ActiveActionSet(
                action_set=context.default_action_set,
                subaction_path=xr.NULL_PATH,
            )
            xr.sync_actions(
                session=session,
                sync_info=xr.ActionsSyncInfo(
                    count_active_action_sets=1,
                    active_action_sets=ctypes.pointer(active_action_set),
                ),
            )

            n_paths = ctypes.c_uint32(0)
            result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
            if xr.check_result(result).is_exception():
                raise result
            vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
            # print(xr.Result(result), n_paths.value)
            result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
            if xr.check_result(result).is_exception():
                raise result
            # print(xr.Result(result), n_paths.value)
            # print(*vive_tracker_paths)
            found_tracker_count = 0
            for index, space in enumerate(tracker_action_spaces):
                space_location = xr.locate_space(
                    space=space,
                    base_space=context.space,
                    time=frame_state.predicted_display_time,
                )
                if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    print(f"{role_strings[index]}: {space_location.pose}")

                    if role_strings[index] == 'right_elbow':
                        if first:
                            visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], 
                                                        [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 0)
                            first = False
                        else:
                            visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], 
                                                    [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 0)
                    elif role_strings[index] == 'left_elbow':
                        if first_2:
                            visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], 
                                                        [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 1)
                            first_2 = False
                        else:
                            visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], 
                                                    [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 1)
                    elif role_strings[index] == 'chest':
                        if first_3:
                            visualizer.set_pose_first([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], 
                                                        [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 2)
                            first_3 = False
                        else:
                            visualizer.set_pose([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z], 
                                                    [space_location.pose.orientation.w, space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z], 2)

                    found_tracker_count += 1
            if found_tracker_count == 0:
                print("no trackers found")

        # Slow things down, especially since we are not rendering anything
        # time.sleep(0.5)
        # Don't run forever
        # if frame_index > 50000:
        #     break
    if not session_was_focused:
        print("This OpenXR session never entered the FOCUSED state. Did you wear the headset?")
