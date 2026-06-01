"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates 
while considering the uncertainty of the obstacles. The students should show that the proposed 
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) planning
        2) cmdFirmware

"""

import numpy as np
from collections import deque

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable third-party packages.
try:
    import example_custom_utils as ecu
except ImportError:
    from . import example_custom_utils as ecu

#########################
# REPLACE THIS (END) ####
#########################

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

class Controller():
    """Controller class using an A* search planner for trajectory planning.

    This controller plans a 2D path from the start, through each gate center (following a test sequence),
    and finally to the target hover. General obstacles are inflated.
    In addition, for each gate, we transform grid cells into the gate's local frame (using its yaw) and mark cells
    as obstacles if their lateral coordinate (|local_x|) falls outside a safe corridor.
    This forces the planner to generate a path that passes through the gate center.
    
    Moreover, after each segment is planned by A*, we verify that the discrete path ends near the desired gate center.
    If not, we append the gate center to force the path through that point.
    
    The resulting 2D path is lifted into 3D (constant altitude) and smoothed using PCHIP interpolation.
    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False):
        """
        Initialization of the controller.
        """
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        self.NOMINAL_GATES = initial_info.get("nominal_gates_pos_and_type", [])
        self.NOMINAL_OBSTACLES = initial_info.get("nominal_obstacles_pos", [])

        if use_firmware:
            self.ctrl = None
        else:
            self.ctrl = PIDController()
            self.KF = initial_info["quadrotor_kf"]

        self.reset()
        self.interEpisodeReset()

        t_scaled = self.planning(use_firmware, initial_info)

        # plot_trajectory(
        #     t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z, 
        #     self.ref_vx, self.ref_vy, self.ref_vz, self.ref_ax, self.ref_ay, self.ref_az,
        #     self.NOMINAL_GATES, self.NOMINAL_OBSTACLES)
        # # # draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        self.notify_stop = False
        self.landing = False
        self.stop = False

    def planning(self, use_firmware, initial_info):
        """
        Trajectory planning algorithm using A* search in the XY plane.
        
        The algorithm:
        1. Sets up a grid occupancy map with inflated obstacles and gate-edge markings.
        2. Defines a waypoint sequence: start position, lateral gate entries/exits, and final target.
        3. Plans an A* path for each segment.
        4. Densifies the discrete A* path.
        5. Converts the 2D path to 3D (using a constant altitude) and smooths it with PCHIP.
        
        Returns:
            t_scaled (ndarray): A time vector for the reference trajectory.
        """
        # ----- 1) Grid Setup -----
        x_min, x_max = -3.5, 3.5
        y_min, y_max = -3.5, 3.5
        resolution = 0.1 # grid cell size in meters
        x_coords = np.arange(x_min, x_max + resolution, resolution)
        y_coords = np.arange(y_min, y_max + resolution, resolution)
        grid_map = np.zeros((len(x_coords), len(y_coords)))
        
        obst_radius = 0.3  # inflate obstacles with safety radius (m)
        for i, xx in enumerate(x_coords):
            for j, yy in enumerate(y_coords):
                for obst in self.NOMINAL_OBSTACLES:
                    ox, oy = obst[0], obst[1]
                    if np.hypot(xx - ox, yy - oy) <= obst_radius:
                        grid_map[i, j] = 1
                        break


        # ----- 2) Utility Functions for Grid Conversions -----
        def pos_to_idx(pos):
            return ecu.pos_to_idx(pos, x_min, y_min, resolution)
        def idx_to_pos(idx):
            return ecu.idx_to_pos(idx, x_min, y_min, resolution)
        
        # ----- 3) Mark Gate Lateral Edges as Obstacles -----
        # Each gate is 0.4 m wide (half-width = 0.2 m). Define a safe corridor (0.2 m to 0.25 m is considered occupied). )
        for gate in self.NOMINAL_GATES:
            gx, gy, _, _, _, gate_yaw, _ = gate
            thickness_range = np.arange(-0.35, 0.35, 0.01)
            offset_range = np.arange(0.1, 0.26, 0.01)

            for thickness in thickness_range:
                for offset in offset_range:
                    if gate_yaw == 0.0:  # Horizontal gate
                        # if gx == 2.0 and gy == -1.5:
                        #     i_left, j_left = pos_to_idx((gx - offset + 0.2, gy + thickness))
                        #     i_right, j_right = pos_to_idx((gx + offset + 0.2, gy + thickness))
                        # else:
                        i_left, j_left = pos_to_idx((gx - offset, gy + thickness))
                        i_right, j_right = pos_to_idx((gx + offset, gy + thickness))
                        grid_map[i_left, j_left] = 1  # Mark left edge as occupied
                        grid_map[i_right, j_right] = 1  # Mark right edge as occupied
                    else:  # Vertical gate
                        # if gx == 0.5 and gy == -2.5:
                        #     i_top, j_top = pos_to_idx((gx + thickness, gy + offset - 0.2))
                        #     i_bottom, j_bottom = pos_to_idx((gx + thickness, gy - offset - 0.2))
                        # else:
                        i_top, j_top = pos_to_idx((gx + thickness, gy + offset))
                        i_bottom, j_bottom = pos_to_idx((gx + thickness, gy - offset))
                        grid_map[i_top, j_top] = 1  # Mark top edge as occupied
                        grid_map[i_bottom, j_bottom] = 1  # Mark bottom edge as occupied

        # Plot the grid map for debugging.
        # ecu.plot_grid_map(grid_map, x_min, x_max, y_min, y_max, resolution)

        # ----- 4) Define the Waypoint Sequence -----
        start_pos = (self.initial_obs[0], self.initial_obs[2])
        # test_sequence = [0, 1, 0, 2, 1, 3]
        test_sequence = [0, 2, 3, 1, 0, 3]
        gate_centers = [(self.NOMINAL_GATES[i][0], self.NOMINAL_GATES[i][1]) for i in test_sequence]
        final_target = (initial_info["x_reference"][0], initial_info["x_reference"][2])
        
        # For each gate, we generate an entry and exit point.
        entry_exit_offset = 0.3  # offset for entry/exit points
        exits = []
        gates_exited = []
        gate_entries_exits = []
        previous_gate_exit = start_pos
        for gate in gate_centers:
            # Here, adjust offsets based on gate orientation if needed.
            # For example, shift in x by +/- 0.15 m.
            if gate[0] == 0.0:
                entry = (gate[0] - entry_exit_offset, gate[1])
                exit  = (gate[0] + entry_exit_offset, gate[1])
            elif gate[0] == 0.5:
                entry = (gate[0] - entry_exit_offset, gate[1])
                exit  = (gate[0] + entry_exit_offset, gate[1])
            elif gate[0] == 2.0:
                entry = (gate[0], gate[1] - entry_exit_offset)
                exit  = (gate[0], gate[1] + entry_exit_offset)
            else:
                entry = (gate[0], gate[1] - entry_exit_offset)
                exit  = (gate[0], gate[1] + entry_exit_offset)
            if ecu.distance(previous_gate_exit, exit) < ecu.distance(previous_gate_exit, entry): 
                # If the exit is closer to the previous gate exit, use it.
                entry, exit = exit, entry
            # Keep track of the last exit for the next gate.
            previous_gate_exit = exit
            exits.append(exit)
            gates_exited.append(gate)
            gate_entries_exits.extend([entry, gate, exit])
        # Construct the full sequence.
        full_sequence = [start_pos] + gate_entries_exits + [final_target]

        # ----- 5) Plan A* Path Segment by Segment -----
        full_path = []
        exit_idx = 0
        gate_marked_as_occupied = False
        for idx in range(len(full_sequence) - 1):
            seg_start = full_sequence[idx]
            seg_goal = full_sequence[idx + 1]
            start_idx = pos_to_idx(seg_start)
            goal_idx = pos_to_idx(seg_goal)

            if gate_marked_as_occupied:
                gate = gates_exited[exit_idx - 1]
                # Mark the gate as occupied in the grid map.
                if gate[0] in [2.0, -0.5]:
                    for offset in np.arange(-0.19, 0.20, 0.01):  # mark range 0.2 to 0.25 as occupied
                        i_gate, j_gate = pos_to_idx((gate[0] + offset, gate[1]))
                        grid_map[i_gate, j_gate] = 0  # mark gate as unoccupied 
                else:
                    for offset in np.arange(-0.19, 0.20, 0.01):  # mark range 0.2 to 0.25 as occupied
                        i_gate, j_gate = pos_to_idx((gate[0], gate[1] + offset))
                        grid_map[i_gate, j_gate] = 0  # mark gate as unoccupied 
                gate_marked_as_occupied = False
                # ecu.plot_grid_map(grid_map, x_min, x_max, y_min, y_max, resolution)

            # Mark the 0.4 m wide gate just crossed as occupied.
            if seg_start == exits[exit_idx]:
                gate = gates_exited[exit_idx]
                exit_idx += 1
                # Mark the gate as occupied in the grid map.
                if gate[0] in [2.0, -0.5]:
                    for offset in np.arange(-0.2, 0.2, 0.01):  # mark range 0.2 to 0.25 as occupied
                        i_gate, j_gate = pos_to_idx((gate[0] + offset, gate[1]))
                        grid_map[i_gate, j_gate] = 1  # mark gate as occupied 
                else:
                    for offset in np.arange(-0.2, 0.2, 0.01):  # mark range 0.2 to 0.25 as occupied
                        i_gate, j_gate = pos_to_idx((gate[0], gate[1] + offset))
                        grid_map[i_gate, j_gate] = 1  # mark gate as occupied 
                gate_marked_as_occupied = True
                # ecu.plot_grid_map(grid_map, x_min, x_max, y_min, y_max, resolution)

            path_indices = ecu.astar_search(start_idx, goal_idx, grid_map)
            if path_indices is None:
                print("A* failed to find a path from", seg_start, "to", seg_goal)
                continue
            path_segment = [idx_to_pos(idd) for idd in path_indices]
            # Ensure the segment ends at the goal.
            if np.hypot(path_segment[-1][0]-seg_goal[0], path_segment[-1][1]-seg_goal[1]) > 0.05:
                path_segment.append(seg_goal)
            # Densify the segment to add intermediate waypoints.
            path_segment = ecu.densify_path(path_segment, resolution=0.2)
            if full_path and full_path[-1] == path_segment[0]:
                full_path.extend(path_segment[1:])
            else:
                full_path.extend(path_segment)

            

        # Remove duplicate consecutive points.
        def remove_duplicates(path):
            cleaned = [path[0]]
            for p in path[1:]:
                if np.linalg.norm(np.array(p) - np.array(cleaned[-1])) > 1e-3:
                    cleaned.append(p)
            return cleaned
        full_path = remove_duplicates(full_path)


        # ----- 6) Convert 2D Path to 3D and Smooth the Trajectory -----
        z_constant = 1.0  # constant altitude for flight
        waypoints = [[pt[0], pt[1], z_constant] for pt in full_path]
        self.waypoints = np.array(waypoints)
        # Smooth the trajectory using PCHIP interpolation.
        (
            ref_x, ref_y, ref_z,
            ref_vx, ref_vy, ref_vz,
            ref_ax, ref_ay, ref_az,
            t_scaled
        ) = ecu.pchip_smooth(waypoints, self.CTRL_FREQ, v_max_xy=1.5, a_max_xy=10.0)
        self.ref_x, self.ref_y, self.ref_z = ref_x, ref_y, ref_z
        self.ref_vx, self.ref_vy, self.ref_vz = ref_vx, ref_vy, ref_vz
        self.ref_ax, self.ref_ay, self.ref_az = ref_ax, ref_ay, ref_az
        return t_scaled

    def cmdFirmware(self, time, obs, reward=None, done=None, info=None):
        """Generate firmware commands for the Crazyflie via CrazySwarm."""
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] 'cmdFirmware' used but Controller was created with 'use_firmware' = False.")
        iteration = int(time * self.CTRL_FREQ)
        offset = int(1 * self.CTRL_FREQ)  # delay after takeoff
        L = len(self.ref_x)
        if iteration == 0:
            height = 1.0
            duration = 2.0
            return Command.TAKEOFF, [height, duration]
        elif iteration >= offset and iteration < offset + L:
            step = iteration - offset
            target_pos = np.array([self.ref_x[step], self.ref_y[step], 1.0])
            target_vel = np.array([self.ref_vx[step], self.ref_vy[step], 0.0])
            target_acc = np.array([self.ref_ax[step], self.ref_ay[step], 0.0])
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)
            if step == len(self.ref_x) - 1:
                self.notify_stop = True
            return Command.FULLSTATE, [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        elif iteration >= offset + L and self.notify_stop:
            self.landing = True
            self.notify_stop = False
            return Command.NOTIFYSETPOINTSTOP, []
        elif iteration >= offset + L + 1 and self.landing:
            height = 0.0
            duration = 3.0
            self.stop = True
            self.landing = False
            return Command.LAND, [height, duration]
        elif iteration >= offset + L + 3 * self.CTRL_FREQ and self.stop:
            self.stop = False
            return Command.STOP, []
        else:
            return Command.NONE, []

    def cmdSimOnly(self, time, obs, reward=None, done=None, info=None):
        """Software-only control for simulation mode using PID control."""
        if self.ctrl is None:
            raise RuntimeError("[ERROR] 'cmdSimOnly' used but Controller was created with 'use_firmware' = True.")
        iteration = int(time * self.CTRL_FREQ)
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        return target_p, target_v

    def reset(self):
        """Reset data buffers and counters."""
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.interstep_counter = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        """Reset timing variables between episodes."""
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0

