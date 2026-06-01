import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
import csv
import matplotlib.pyplot as plt
import os
import random

# Camera intrinsics (3x3)
K = np.array([
    [698.86,   0.0,   306.91],
    [  0.0,  699.13, 150.34],
    [  0.0,    0.0,    1.0 ]
], dtype=np.float64)

# Distortion coefficients
distCoeffs = np.array([0.191887, -0.563680, -0.003676, -0.002037, 0.0], dtype=np.float64)

# Extrinsic transform T_{CB} (4x4) from Body frame {B} to Camera frame {C}
T_CB = np.array([
    [ 0.0, -1.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0]
], dtype=np.float64)

def quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z):
    """
    Convert quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    norm = math.sqrt(q_w*q_w + q_x*q_x + q_y*q_y + q_z*q_z)
    w, x, y, z = q_w/norm, q_x/norm, q_y/norm, q_z/norm

    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)
    return R

def build_homogeneous_transform(R, t):
    """
    Build a 4x4 homogeneous transform from a 3x3 rotation and 3x1 translation:
         [ R  t ]
         [ 0  1 ]
    """
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3]   = t.flatten()
    return T

def intersect_ray_with_ground(camera_center, direction):
    """
    Intersect the ray (camera_center + t*direction) with z=0 plane in inertial frame.
    Returns (X, Y) or None if no valid intersection (e.g. direction.z ~ 0 or behind camera).
    """
    cz = camera_center[2]
    dz = direction[2]
    if abs(dz) < 1e-12:
        return None  # Ray is parallel to the ground plane

    t = -cz / dz
    if t < 0:
        return None  # Intersection is behind or camera is below ground

    X = camera_center[0] + t*direction[0]
    Y = camera_center[1] + t*direction[1]
    return (X, Y)

def ransac_2d_points(points, thresh=0.15, iterations=300):
    """
    A minimal 2D RANSAC: pick 1 random point as seed, gather inliers within 'thresh' distance.
    Returns the best inlier set and the average (X, Y) of that set.
    """
    if len(points) < 3:
        return points, np.mean(points, axis=0)

    best_inliers = []
    best_center = None
    for _ in range(iterations):
        seed_idx = random.randrange(len(points))
        seed_pt = points[seed_idx]
        dists = np.linalg.norm(points - seed_pt, axis=1)
        inliers = points[dists < thresh]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_center = np.mean(inliers, axis=0)
    if best_center is None:
        best_center = np.mean(points, axis=0)
        best_inliers = points
    return best_inliers, best_center

# MAIN PIPELINE

def main():
    """
    Detect multiple green circles in each frame, project them to inertial coords,
    cluster them into 6 groups (for 6 floor targets), apply 2D RANSAC, and print
    the final 6 positions.
    """
    # File paths (adjust as needed)
    csv_path = "lab3_pose.csv"
    image_folder = "output_folder"

    # Tuning parameters for color thresholding & morphological filtering

    lower_green = (30, 60, 20)
    upper_green = (80, 255, 95)

    kernel_size = 5

    # Contour acceptance
    min_area_threshold = 200
    fill_ratio_threshold = 0.5

    # RANSAC for final cluster refinement
    ransac_thresh = 0.15
    num_iterations = 300

    # Load drone poses from CSV
    poses = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 8:
                continue
            idx = int(row[0])
            p_x = float(row[1])
            p_y = float(row[2])
            p_z = float(row[3])
            q_w = float(row[4])
            q_x_ = float(row[5])
            q_y_ = float(row[6])
            q_z_ = float(row[7])
            poses[idx] = (p_x, p_y, p_z, q_w, q_x_, q_y_, q_z_)

    # For storing 2D ground-plane detections across all frames
    # Each detection is a row: (X, Y)
    all_detections = []

    last_idx = 2892
    # "Real-time" loop: process each image
    for idx in range(last_idx):
        img_file = os.path.join(image_folder, f"image_{idx}.jpg")

        if idx not in poses:
            continue
        p_x, p_y, p_z, q_w, q_x_, q_y_, q_z_ = poses[idx]

        # Load image
        img_bgr = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue  # Could not read

        # Undistort
        img_undist = cv2.undistort(img_bgr, K, distCoeffs)

        # Convert to HSV
        img_hsv = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HSV)

        # Threshold for green color
        mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

        # Morphological filtering
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour that looks like a circle, do the pinhole + transform
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_threshold:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circle_area = math.pi * (radius**2)
            fill_ratio = area / (circle_area + 1e-9)

            if fill_ratio < fill_ratio_threshold:
                # Not round enough
                continue

            # We have a valid circle
            # Draw detection for visualization
            cv2.circle(img_undist, (int(cx), int(cy)), int(radius), (0,0,255), 2)
            cv2.circle(img_undist, (int(cx), int(cy)), 2, (0,0,255), -1)

            # Convert pixel -> direction in camera frame
            uv_hom = np.array([cx, cy, 1.0], dtype=np.float64)
            invK = np.linalg.inv(K)
            dir_cam = invK @ uv_hom

            # Build T_{IB}
            R_IB = quaternion_to_rotation_matrix(q_w, q_x_, q_y_, q_z_)
            t_IB = np.array([p_x, p_y, p_z], dtype=np.float64)
            T_IB = build_homogeneous_transform(R_IB, t_IB)

            # Compose T_{IC} = T_{IB} * T_{CB}^{-1}
            # Since T_{CB}^{-1} = T_{CB}, we can just multiply directly
            T_IC = T_IB @ T_CB
            R_IC = T_IC[0:3, 0:3]
            cam_center_inertial = T_IC[0:3, 3]

            # direction in inertial
            dir_inertial = R_IC @ dir_cam

            # intersect with ground plane z=0
            XY = intersect_ray_with_ground(cam_center_inertial, dir_inertial)
            if XY is not None:
                X, Y = XY
                all_detections.append((X, Y))

        # Show the detection + mask side by side
        mask_vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        combined_vis = np.hstack((img_undist, mask_vis))
        cv2.imshow("Detection (left) & Mask (right)", combined_vis)

        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):
            break
        
        print(f"Processed frame {idx}/{last_idx}", end='\r')

    # Close display windows
    cv2.destroyAllWindows()

    # CLUSTER INTO 6 GROUPS & REFINE WITH RANSAC
    detections_np = np.array(all_detections, dtype=np.float64)
    if len(detections_np) < 6:
        print(f"Not enough detections to identify 6 distinct targets. Found {len(detections_np)} total points.")
        return


    # Use KMeans to group detections into 6 clusters
    print("Using KMeans clustering for 6 targets...")
    kmeans = KMeans(n_clusters=6, random_state=42).fit(detections_np)
    labels = kmeans.labels_

    final_positions = []
    for cluster_id in range(6):
        cluster_points = detections_np[labels == cluster_id]
        if len(cluster_points) < 3:
            # not enough points in this cluster to do RANSAC
            # fallback: average
            if len(cluster_points) == 0:
                # no points, skip
                continue
            mean_pt = np.mean(cluster_points, axis=0)
            final_positions.append(mean_pt)
        else:
            # 2D RANSAC for outlier rejection in each cluster
            _, best_center = ransac_2d_points(cluster_points, thresh=ransac_thresh, iterations=num_iterations)
            final_positions.append(best_center)

    # Print final 6 cluster centers
    print("====================================")
    print("Lab3 Final 6 Target Localizations (RANSAC + Clustering)")
    for i, pos in enumerate(final_positions, start=1):
        print(f"Target #{i}: (X, Y) = ({pos[0]:.9f}, {pos[1]:.9f})")
    print("====================================")

    

    # Plot all detections
    plt.scatter(detections_np[:, 0], detections_np[:, 1], c='blue', label='All Detections', s=10)

    # Plot final positions
    final_positions_np = np.array(final_positions, dtype=np.float64)
    plt.scatter(final_positions_np[:, 0], final_positions_np[:, 1], c='red', label='Final Positions', s=50, marker='x')

    # Label the plot
    plt.title('Detected Positions and Final Target Localizations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_positions.png')
    plt.show()

    # Save all detections to a CSV file
    detections_file = "all_detections.csv"
    np.savetxt(detections_file, detections_np, delimiter=",", header="X,Y", comments="")
    print(f"All detections saved to {detections_file}")

    # Save final positions to a CSV file
    final_positions_file = "final_positions.csv"
    np.savetxt(final_positions_file, final_positions_np, delimiter=",", header="X,Y", comments="")
    print(f"Final positions saved to {final_positions_file}")

if __name__ == "__main__":
    main()