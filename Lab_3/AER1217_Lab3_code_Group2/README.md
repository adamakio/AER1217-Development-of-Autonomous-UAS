**README: Lab3 Target Detection & Localization**

**Overview**  
This Python script uses OpenCV to detect **green circles** on the ground in images captured by a Parrot AR.Drone’s downward‐facing camera. It then **projects** those 2D detections into the **inertial (Vicon) frame** using the drone’s pose and camera parameters, clusters the resulting 3D points into **6 groups** (for 6 physical targets), and refines each target’s location with a simple **RANSAC** step. Finally, it prints out the \((X,Y)\) coordinates for each of the **six** targets.

---

## 1. File Organization

1. **CSV Pose File** (`lab3_pose.csv`):  
   Contains drone pose data (position + quaternion) at each image index.  
   Columns:  
   ```
   idx, p_x, p_y, p_z, q_w, q_x, q_y, q_z
   ```
2. **Image Folder** (`output_folder`):  
   Contains images named like `image_XXXX.jpg`, one per CSV index.

3. **This Script**:  
   - Reads the CSV, loads images in ascending order, and processes them in a loop.  
   - At the end, clusters all detections and applies RANSAC to find each target’s final coordinates.

---

## 2. Code Sections

### 2.1 Imports and Global Parameters

```python
import cv2, numpy as np, math, csv, glob, os, random
from sklearn.cluster import KMeans
```
- **cv2** (OpenCV) for image operations.  
- **numpy** for array math.  
- **sklearn.cluster** (KMeans) for clustering.  
- **math**, **csv**, **glob**, **os**, **random** for standard Python utilities.

```python
# Camera intrinsics (K) and distortion (distCoeffs)
# Extrinsic T_CB from Body frame to Camera frame
```
- These constants come from the lab calibration.  
- `K` is a 3×3 matrix for the pinhole camera model.  
- `distCoeffs` is the lens‐distortion vector.  
- `T_CB` is a 4×4 homogeneous transform from the drone’s body frame to the camera frame.

### 2.2 Helper Functions

1. **`quaternion_to_rotation_matrix`**  
   - Converts a quaternion \((w,x,y,z)\) to a 3×3 rotation matrix, normalizing the quaternion first.

2. **`build_homogeneous_transform`**  
   - Builds a 4×4 transform from a 3×3 rotation and 3×1 translation vector:
     \[
       T = 
       \begin{bmatrix}
         R & t \\
         0 & 1
       \end{bmatrix}.
     \]

3. **`intersect_ray_with_ground`**  
   - Takes a **camera center** \(\mathbf{c}\) and a **direction** \(\mathbf{d}\) in 3D (both in the inertial frame) and computes where that ray intersects the plane \(z=0\).  
   - Returns \((X,Y)\) or `None` if it’s invalid (ray parallel or behind camera).

4. **`ransac_2d_points`**  
   - A minimal 2D RANSAC function to find an inlier set around a random “seed” point, used to robustly find a center from multiple scattered points.

### 2.3 Main Pipeline (`main()`)

1. **Paths & Parameters**  
   ```python
   csv_path = ".../lab3_pose.csv"
   image_folder = ".../output_folder"
   ```
   - Adjust these to match your file locations.

   **HSV Threshold**  
   ```python
   lower_green = (30, 60, 20)
   upper_green = (80, 255, 95)
   ```
   - Defines the color range in HSV to isolate green circles.

   **Morphological Filter**  
   ```python
   kernel_size = 5
   ```
   - Used to remove small noise in the binary mask.

   **Contour Acceptance**  
   ```python
   min_area_threshold = 200
   fill_ratio_threshold = 0.5
   ```
   - Ensures the detected contour is large enough (`area > 200`) and “round enough” (`fill_ratio > 0.5`).

   **RANSAC**  
   ```python
   ransac_thresh = 0.15
   num_iterations = 300
   ```
   - For final outlier rejection in each cluster.

2. **Load Drone Poses**  
   ```python
   poses = {}
   with open(csv_path, 'r') as f:
       ...
       poses[idx] = (p_x, p_y, p_z, q_w, q_x_, q_y_, q_z_)
   ```
   - Reads each line of the CSV, storing the drone’s position + quaternion in a dictionary keyed by `idx`.

3. **Collect Images**  
   ```python
   image_files = sorted(glob.glob(os.path.join(image_folder, "image_*.jpg")))
   ```
   - Finds all `image_XXXX.jpg` files, sorted by name.

4. **Loop Over Images**  
   ```python
   for img_file in image_files:
       ...
   ```
   - Extract `idx` from the filename, look up the drone pose in `poses[idx]`.

5. **Undistort & HSV Threshold**  
   ```python
   img_undist = cv2.undistort(img_bgr, K, distCoeffs)
   img_hsv = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HSV)
   mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
   ```
   - Removes lens distortion, converts to HSV, and creates a binary mask where green is white (255).

6. **Morphological Operations**  
   ```python
   kernel = np.ones((kernel_size, kernel_size), np.uint8)
   mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
   mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
   ```
   - Removes small noise and fills small holes.

7. **Find Contours**  
   ```python
   contours, _ = cv2.findContours(mask_clean, ...)
   for cnt in contours:
       ...
   ```
   - Each contour is a potential green circle. We compute:
     - `area = cv2.contourArea(cnt)`
     - `(cx, cy), radius = cv2.minEnclosingCircle(cnt)`
     - `fill_ratio = area / circle_area`

8. **Filter Valid Circles**  
   - Skip if `area < min_area_threshold` or `fill_ratio < fill_ratio_threshold`.  
   - If valid, **draw** a red circle on `img_undist` for visualization.

9. **Back‐Projection & Pose Composition**  
   ```python
   uv_hom = np.array([cx, cy, 1.0])
   dir_cam = invK @ uv_hom  # direction in camera coords

   # Build T_{IB} from CSV
   R_IB = quaternion_to_rotation_matrix(q_w, q_x_, q_y_, q_z_)
   t_IB = [p_x, p_y, p_z]
   T_IB = build_homogeneous_transform(R_IB, t_IB)

   # T_{IC} = T_{IB} * T_{CB}
   T_IC = T_IB @ T_CB
   R_IC = T_IC[0:3, 0:3]
   cam_center_inertial = T_IC[0:3, 3]

   dir_inertial = R_IC @ dir_cam
   XY = intersect_ray_with_ground(cam_center_inertial, dir_inertial)
   if XY is not None:
       all_detections.append(XY)
   ```
   - Converts the **pixel** location \((cx,cy)\) into a **ray** in the camera frame, then rotates that ray into the inertial frame.  
   - Intersects with the ground plane \(z=0\) to get \((X,Y)\) in inertial coordinates.  
   - Accumulates these detections in `all_detections`.

10. **Display**  
   ```python
   mask_vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
   combined_vis = np.hstack((img_undist, mask_vis))
   cv2.imshow("Detection (left) & Mask (right)", combined_vis)
   ```
   - Shows the original detection overlay (left) and the binary mask (right). Press **ESC** or **q** to quit early.

11. **After the Loop**  
   ```python
   detections_np = np.array(all_detections)
   if len(detections_np) < 6:
       ...
   kmeans = KMeans(n_clusters=6, random_state=42).fit(detections_np)
   ...
   ```
   - If we have enough points, we cluster them into **6** groups. Each group corresponds to one of the 6 floor targets.

12. **RANSAC**  
   ```python
   for cluster_id in range(6):
       cluster_points = detections_np[labels == cluster_id]
       if len(cluster_points) < 3:
           ...
       else:
           _, best_center = ransac_2d_points(cluster_points, ...)
           final_positions.append(best_center)
   ```
   - For each cluster, we do a 2D RANSAC to remove outliers and refine the cluster’s final \((X,Y)\).  
   - We print the final 6 target locations.

---

## 3. How to Run

1. **Install Dependencies**:  
   ```bash
   pip install opencv-python numpy scikit-learn matplotlib
   ```
2. **Check File Paths**:  
   - Update `csv_path` and `image_folder` if necessary.
3. **Run**:  
   ```bash
   python your_script.py
   ```
4. **View Results**:  
   - A window shows “Detection (left) & Mask (right)”.  
   - Press **ESC** or **q** to stop early.  
   - The terminal prints the final 6 cluster centers.
   - A plot of the final 6 points is shown.

---

## 4. Further Adjustments

- **HSV Threshold** (`lower_green`, `upper_green`): Tune these if lighting or color changes.  
- **Morphological Kernel** (`kernel_size`): Adjust if you see too much noise or if you lose parts of the circle.  
- **min_area_threshold** / **fill_ratio_threshold**: Adjust for circle size and shape.  
- **ransac_thresh**: If your final points are scattered more/less, tweak the inlier distance threshold.  

---

## 5. Conclusion

This code implements a **complete pipeline** to detect green circles, back‐project them into 3D, cluster them into 6 known targets, and refine each target’s location via RANSAC. It satisfies the lab requirement to **use multiple images** for robust detection, and leverages **OpenCV** for image processing and **k‐means** + **RANSAC** for outlier rejection.

**End of README**