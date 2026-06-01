import numpy as np
from heapq import heappush, heappop
from scipy.interpolate import interp1d, BSpline, splrep
import matplotlib.pyplot as plt

def plot_grid_map(grid_map, x_min, x_max, y_min, y_max, resolution):
    """
    Visualize the occupancy grid map.
    
    Args:
        grid_map (ndarray): 2D array with 0 for free and 1 for occupied cells.
        x_min (float): Minimum x-coordinate of the grid.
        x_max (float): Maximum x-coordinate of the grid.
        y_min (float): Minimum y-coordinate of the grid.
        y_max (float): Maximum y-coordinate of the grid.
        resolution (float): Grid cell size in meters.
    
    The function uses imshow to display the grid. The 'origin' is set to 'lower' so that the
    lower left corner corresponds to (x_min, y_min).
    """
    plt.figure(figsize=(8,8))
    extent = [x_min, x_max, y_min, y_max]
    # Transpose grid_map so that the first dimension maps to x-axis
    plt.imshow(grid_map.T, origin='lower', extent=extent, cmap='Greys', interpolation='nearest')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Occupancy Grid Map')
    plt.colorbar(label='Occupancy')
    plt.show()

# ----- Grid Conversion Helpers -----
def pos_to_idx(pos, x_min, y_min, resolution):
    """Convert world position (x, y) to grid indices."""
    px, py = pos
    i = int(round((px - x_min) / resolution))
    j = int(round((py - y_min) / resolution))
    return (i, j)

def idx_to_pos(idx, x_min, y_min, resolution):
    """Convert grid indices to world position (x, y)."""
    i, j = idx
    xx = x_min + i * resolution
    yy = y_min + j * resolution
    return (xx, yy)

def astar_search(start_idx, goal_idx, grid):
    """
    Perform A* search on a grid.
    
    Returns:
        A list of grid indices (tuples) from start to goal, or None if no path is found.
    """

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    open_set = []
    # f = g + heuristic, g is cost from start
    heuristic = np.hypot(goal_idx[0] - start_idx[0], goal_idx[1] - start_idx[1])
    heappush(open_set, (heuristic, 0, start_idx))
    
    came_from = {}         # For path reconstruction.
    g_score = {start_idx: 0}  # Cost from start to each node.
    closed_set = set()     # Expanded nodes.
    
    while open_set:
        f, curr_g, current = heappop(open_set)
        if current in closed_set:
            continue
        
        # Goal test.
        if current == goal_idx:
            # Reconstruct the path.
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        closed_set.add(current)
        
        # Expand neighbors.
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            # Check grid boundaries.
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue  # Skip occupied cells.
                tentative_g = curr_g + np.hypot(dx, dy)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = np.hypot(goal_idx[0] - neighbor[0], goal_idx[1] - neighbor[1])
                    heappush(open_set, (tentative_g + h, tentative_g, neighbor))
    return None  # No path found.


# ----- Path Densification Helper -----
def densify_path(path, resolution=0.02):
    """
    Densify a 2D path (list of (x,y) tuples) by interpolating between points.
    Returns a list of (x,y) points with roughly the given resolution.
    """
    path = np.array(path)
    # Compute cumulative distances along the path.
    dists = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    dists = np.insert(dists, 0, 0.0)
    total_length = dists[-1]
    num_points = max(int(total_length / resolution), 2)
    interp_t = np.linspace(0, total_length, num_points)
    fx = interp1d(dists, path[:, 0])
    fy = interp1d(dists, path[:, 1])
    densified = np.vstack((fx(interp_t), fy(interp_t))).T
    return densified.tolist()


# ----- Trajectory Smoothing with CubicSpline -----
def pchip_smooth(waypoints, freq,
                 v_max_xy=1.5, a_max_xy=4.0,
                 v_max_z =1.0, a_max_z =3.0,
                 margin=1.05):
    """
    Smooth a 3‑D waypoint list with a natural cubic spline and generate
    position, velocity, acceleration references that automatically slow
    down in high‑curvature zones.

    Parameters
    ----------
    waypoints  : array‑like shape (N,3)
    freq       : controller rate [Hz]
    v_max_xy   : horizontal speed limit  (m s‑1)
    a_max_xy   : horizontal accel limit  (m s‑2)
    v_max_z    : vertical   speed limit  (m s‑1)
    a_max_z    : vertical   accel limit  (m s‑2)
    margin     : safety factor (>1). 1.05 = 5 % cushion.

    Returns
    -------
    (ref_x, ref_y, ref_z,          -- position  arrays
     ref_vx, ref_vy, ref_vz,       -- velocity  arrays
     ref_ax, ref_ay, ref_az,       -- accel     arrays
     t_scaled)                     -- time vector [s]
    """

    # ---------- 1.  Geometry – spline in arc‑length s ----------------------
    wp = np.asarray(waypoints, float)
    seg_len = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    s_knots = np.concatenate(([0.0], np.cumsum(seg_len)))
    L_total = s_knots[-1]

    # ---------- 2.  Fit cubic B‑splines (k=3, s=0 => interpolating) -------
    # tck = (knot vector t, coefficients c, order k)
    tck_x = splrep(s_knots, wp[:, 0], k=3, s=0.1)
    tck_y = splrep(s_knots, wp[:, 1], k=3, s=0.1)
    tck_z = splrep(s_knots, wp[:, 2], k=3, s=0.1)

    cs_x = BSpline(*tck_x)          # BSpline(t, c, k)
    cs_y = BSpline(*tck_y)
    cs_z = BSpline(*tck_z)
    # ---------- 2.  Initial dense sampling in s ---------------------------
    # Start with nominal constant‑speed samples (0.02 m spacing or finer)
    ds_nom   = 0.02
    num_nom  = max(int(L_total / ds_nom) + 1, 3)
    s_nom    = np.linspace(0.0, L_total, num_nom)

    # Position and derivatives wrt s
    x  = cs_x(s_nom);  y  = cs_y(s_nom);  z  = cs_z(s_nom)
    dx = cs_x.derivative(1)(s_nom);  dy = cs_y.derivative(1)(s_nom)
    dz = cs_z.derivative(1)(s_nom)
    ddx = cs_x.derivative(2)(s_nom); ddy = cs_y.derivative(2)(s_nom)
    ddz = cs_z.derivative(2)(s_nom)

    # ---------- 3.  Curvature & per‑point speed cap -----------------------
    # κ = |x'y'' – y'x''| / (x'^2 + y'^2)^(3/2)
    eps = 1e-6
    denom = (dx**2 + dy**2)**1.5 + eps
    kappa = np.abs(dx * ddy - dy * ddx) / denom

    v_cap_curve = np.sqrt(a_max_xy / (kappa + eps))       # from   a = v² κ
    v_cap_xy    = np.minimum(v_cap_curve, v_max_xy)       # also obey v_max_xy

    # ---------- 4.  Build a non‑uniform time mapping t(s) -----------------
    ds = np.diff(s_nom)                      # constant ds_nom
    v_mid = (v_cap_xy[:-1] + v_cap_xy[1:]) * 0.5
    dt_seg = ds / v_mid                     # dt for each segment
    t_nom  = np.zeros_like(s_nom)
    t_nom[1:] = np.cumsum(dt_seg)           # cumulative time
    T_total = t_nom[-1]

    # ---------- 5.  Uniform re‑sampling at controller rate ----------------
    num_samples = int(T_total * freq) + 1
    t_scaled = np.linspace(0.0, T_total, num_samples)
    s_scaled = np.interp(t_scaled, t_nom, s_nom)

    # Position on uniform time grid
    ref_x = cs_x(s_scaled); ref_y = cs_y(s_scaled); ref_z = cs_z(s_scaled)

    # Derivatives wrt s on that grid
    dx_s  = cs_x.derivative(1)(s_scaled)
    dy_s  = cs_y.derivative(1)(s_scaled)
    dz_s  = cs_z.derivative(1)(s_scaled)
    ddx_s = cs_x.derivative(2)(s_scaled)
    ddy_s = cs_y.derivative(2)(s_scaled)
    ddz_s = cs_z.derivative(2)(s_scaled)

    # First & second derivative of s(t) via finite diff
    s_dot  = np.gradient(s_scaled, t_scaled)
    s_ddot = np.gradient(s_dot,   t_scaled)

    # Velocity & acceleration in Cartesian coords
    ref_vx = dx_s * s_dot
    ref_vy = dy_s * s_dot
    ref_vz = dz_s * s_dot

    ref_ax = ddx_s * s_dot**2 + dx_s * s_ddot
    ref_ay = ddy_s * s_dot**2 + dy_s * s_ddot
    ref_az = ddz_s * s_dot**2 + dz_s * s_ddot

    # ---------- 6.  Global safety pass for remaining limits ---------------
    # We may need more than one stretch, but we cap it to max_iter = 4
    max_iter = 4
    itr      = 0
    while True:
        v_xy = np.hypot(ref_vx, ref_vy)
        a_xy = np.hypot(ref_ax, ref_ay)
        scale = max(v_xy.max()/(v_max_xy / margin),
                    abs(ref_vz).max()/(v_max_z / margin),
                    a_xy.max()/(a_max_xy / margin),
                    abs(ref_az).max()/(a_max_z / margin))

        if scale <= 1.0 + 1e-3 or itr >= max_iter:
            break   # trajectory is safe (or we’ve already tried enough)

        # --- stretch the whole timeline and recompute derivatives ---------
        itr     += 1
        t_nom *= scale            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        T_total = t_nom[-1]       # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        num_samples = int(T_total * freq) + 1
        t_scaled = np.linspace(0.0, T_total, num_samples)
        s_scaled = np.interp(t_scaled, t_nom, s_nom)

        # Re‑evaluate derivatives on stretched time base
        dx_s  = cs_x.derivative(1)(s_scaled)
        dy_s  = cs_y.derivative(1)(s_scaled)
        dz_s  = cs_z.derivative(1)(s_scaled)
        ddx_s = cs_x.derivative(2)(s_scaled)
        ddy_s = cs_y.derivative(2)(s_scaled)
        ddz_s = cs_z.derivative(2)(s_scaled)

        s_dot  = np.gradient(s_scaled, t_scaled)
        s_ddot = np.gradient(s_dot,   t_scaled)

        ref_x, ref_y, ref_z = cs_x(s_scaled), cs_y(s_scaled), cs_z(s_scaled)
        ref_vx = dx_s * s_dot
        ref_vy = dy_s * s_dot
        ref_vz = dz_s * s_dot
        ref_ax = ddx_s * s_dot**2 + dx_s * s_ddot
        ref_ay = ddy_s * s_dot**2 + dy_s * s_ddot
        ref_az = ddz_s * s_dot**2 + dz_s * s_ddot

    # ---------- 7.  Done ---------------------------------------------------
    return (ref_x, ref_y, ref_z,
            ref_vx, ref_vy, ref_vz,
            ref_ax, ref_ay, ref_az,
            t_scaled)


# ----- Distance Calculation -----
def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

