=======================
  README for Lab 4 VO
=======================

This repository contains two main files:

1) stereo_vo_base.py
2) lab4.py

They implement a stereo visual odometry pipeline using KITTI dataset images.

-----------------------------------------------------
1. stereo_vo_base.py
-----------------------------------------------------
- This file provides:
  (A) The StereoCamera class:
      - Stores stereo camera parameters:
        baseline (B), focal length (f_len),
        principal point offsets (cu, cv), etc.
      - Example usage:
         cam = StereoCamera(0.537, 721.5377, 721.5377, 721.5377, 609.5593, 172.8540)

  (B) The VisualOdometry class:
      - Manages the entire VO pipeline:
        * feature detection (SIFT)
        * stereo matching & filtering
        * RANSAC-based 3D alignment (pose_estimation)
        * finite-state machine handling (first frame, second frame, subsequent frames)
        
      - Notable methods:

        i) feature_detection(img):
           - Uses SIFT to detect and compute keypoints and descriptors.
           - Returns (kp, des, feature_image).

        ii) featureTracking(prev_kp, cur_kp, img, color, alpha):
           - Visual aid function to draw lines from old to new points in an image.

        iii) find_feature_correspondences(...):
           - Matches features across four images:
             (prev-left, prev-right, current-left, current-right).
           - Filters via stereo epipolar constraints (vertical pixel difference)
             and disparity thresholds (FAR_THRESH, CLOSE_THRESH).
           - Returns an array (N,8) of matched coordinates:
             [prev_left_x, prev_left_y,
              prev_right_x, prev_right_y,
              cur_left_x,  cur_left_y,
              cur_right_x, cur_right_y].

        iv) pose_estimation(features_coor):
           - **Core function** where we:
             1) Triangulate each correspondence into 3D for previous & current frames.
             2) Run RANSAC to reject outliers and estimate a rigid transform (C,r).
             3) Compute the final best alignment (via SVD) using inliers.
             4) Return rotation (3x3), translation (3,), plus the inlier subsets (for plotting).

        v) processFirstFrame / processSecondFrame / processFrame:
           - Each handles a different state of the pipeline (first frame has no previous data, second frame establishes the first transform, subsequent frames do normal tracking).

        vi) update(img_left, img_right, frame_id):
           - Called each iteration with a new stereo image pair.
           - Chooses the correct “process...” method based on frame_stage.
           - Returns the annotated left & right images.

- The script also defines constants:
    STAGE_FIRST_FRAME   (0)
    STAGE_SECOND_FRAME  (1)
    STAGE_DEFAULT_FRAME (2)
  for the pipeline state machine.

-----------------------------------------------------
2. lab4.py
-----------------------------------------------------
- This file is the main driver script:
  (A) Loads ground truth poses from 'ground_truth_pose.mat'.
  (B) Initializes camera parameters (StereoCamera) and VisualOdometry object.
  (C) Reads stereo images from specified paths, in a loop.
  (D) Calls vo.update(...) each frame to get updated rotation & translation.
  (E) Accumulates the local transforms into a global transform, T_hist.
  (F) Converts the estimated camera center pose into the vehicle frame,
      comparing with ground truth.

- Key steps:
  1. For each frame index i:
     - Read left & right images from disk.
     - vo.update(...) returns annotated frames (left with matched features, right with inliers).
     - We form a single “vertical_frame” for side-by-side viewing & writing to 'video_inliers.avi'.
     - Update the global transform T by T_update * T.
  2. We plot the final trajectory (T_vehicle) vs. ground truth, and compute position errors.

- At the end, we print:
    Mean position error: X.XXX m
    Final frame error:   Y.YYY m

  and display a 3D plot comparing VO trajectory vs. the ground-truth path.

-----------------------------------------------------
3. Usage / How to Run
-----------------------------------------------------
- Requirements:
  * Python 3.8+
  * NumPy
  * OpenCV (cv2)
  * SciPy (for sio.loadmat)
  * Matplotlib

- Steps to run:
  1) Place 'ground_truth_pose.mat' and your KITTI image dataset in the appropriate directories (adjust path_* in lab4.py if needed).
  2) Run:  python lab4.py
  3) The script will open a CV window showing the left image on top (with features) and the right image below (with inliers). Press 'q' to exit early.
  4) A 'video_inliers.avi' file is saved, capturing the above display.
  5) A final 3D plot is displayed at the end, with ground truth (red) vs. VO trajectory (blue).
  6) The console prints mean and final position errors.

-----------------------------------------------------
4. Notes on Code Structure
-----------------------------------------------------
- The code uses a “finite-state machine” approach in VisualOdometry:
    STAGE_FIRST_FRAME   -> STAGE_SECOND_FRAME -> STAGE_DEFAULT_FRAME
  so that we detect features in the first pair, then handle the second pair, etc.

- The RANSAC algorithm in pose_estimation uses:
    max_iterations = 2000
    inlier_tolerance = 0.3 (meters)
  to robustly find the best 3D transform.

- The final transform is refined with all inliers using an SVD approach, ensuring a proper rotation with det(UV^T) > 0.

- This pipeline outputs:
    * A real-time window "Visual Odometry (Inliers on Right)"
    * A video file 'video_inliers.avi'
    * 3D trajectory + comparison with ground truth
