"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
import numpy as np
import cv2 as cv
import sys

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

np.random.rand(1217)

class StereoCamera:
    def __init__(self, baseline, focalLength, fx, fy, cu, cv):
        self.baseline = baseline
        self.f_len = focalLength
        self.fx = fx
        self.fy = fy
        self.cu = cu
        self.cv = cv

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)                               # current rotation    (initiated to be eye matrix)
        self.r = np.zeros((3,1))                         # current translation (initiated to be zeros)
        self.kp_l_prev  = None                           # previous key points (left)
        self.des_l_prev = None                           # previous descriptor for key points (left)
        self.kp_r_prev  = None                           # previous key points (right)
        self.des_r_prev = None                           # previoud descriptor key points (right)
        self.detector = cv.xfeatures2d.SIFT_create()     # using sift for detection
        self.feature_color = (255, 191, 0)
        self.inlier_color = (32,165,218)

            
    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img,kp,None)
        return kp, des, feature_image

    def featureTracking(self, prev_kp, cur_kp, img, color=(0,255,0), alpha=0.5):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cover = np.zeros_like(img)
        # Draw the feature tracking 
        for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
            a, b = new.ravel()
            c, d = old.ravel()  
            a,b,c,d = int(a), int(b), int(c), int(d)
            cover = cv.line(cover, (a,b), (c,d), color, 2)
            cover = cv.circle(cover, (a,b), 3, color, -1)
        frame = cv.addWeighted(cover, alpha, img, 0.75, 0)
        
        return frame
    
    def find_feature_correspondences(self, kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r):
        VERTICAL_PX_BUFFER = 1                                # buffer for the epipolor constraint in number of pixels
        FAR_THRESH = 7                                        # 7 pixels is approximately 55m away from the camera 
        CLOSE_THRESH = 65                                     # 65 pixels is approximately 4.2m away from the camera
        
        nfeatures = len(kp_l)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)        # BFMatcher for SIFT or SURF features matching

        ## using the current left image as the anchor image
        match_l_r = bf.match(des_l, des_r)                    # current left to current right
        match_l_l_prev = bf.match(des_l, des_l_prev)          # cur left to prev. left
        match_l_r_prev = bf.match(des_l, des_r_prev)          # cur left to prev. right

        kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
        kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
        kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

        kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
        kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
        kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

        ## loop through all the matched features to find common features
        features_coor = np.zeros((1,8))
        for pt_idx in np.arange(nfeatures):
            if (pt_idx in set(kp_query_idx_l_r)) and (pt_idx in set(kp_query_idx_l_l_prev)) and (pt_idx in set(kp_query_idx_l_r_prev)):
                temp_feature = np.zeros((1,8))
                temp_feature[:, 0:2] = kp_l_prev[kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]].pt 
                temp_feature[:, 2:4] = kp_r_prev[kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]].pt 
                temp_feature[:, 4:6] = kp_l[pt_idx].pt 
                temp_feature[:, 6:8] = kp_r[kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]].pt 
                features_coor = np.vstack((features_coor, temp_feature))
        features_coor = np.delete(features_coor, (0), axis=0)

        ##  additional filter to refine the feature coorespondences
        # 1. drop those features do NOT follow the epipolar constraint
        features_coor = features_coor[
                    (np.absolute(features_coor[:,1] - features_coor[:,3]) < VERTICAL_PX_BUFFER) &
                    (np.absolute(features_coor[:,5] - features_coor[:,7]) < VERTICAL_PX_BUFFER)]

        # 2. drop those features that are either too close or too far from the cameras
        features_coor = features_coor[
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) < CLOSE_THRESH)]

        features_coor = features_coor[
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) < CLOSE_THRESH)]
        # features_coor:
        #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
        return features_coor
    
    def pose_estimation(self, features_coor):
        """
        Estimate the rotation C and translation r (3x1) that align the "previous" camera frame
        to the "current" camera frame, using stereo geometry and RANSAC.

        Parameters
        ----------
        features_coor : (N, 8) ndarray
            Each row = [prev_l_x, prev_l_y, prev_r_x, prev_r_y,
                        cur_l_x,  cur_l_y,  cur_r_x,  cur_r_y]

        Returns
        -------
        C : (3, 3) ndarray
            Estimated rotation matrix from previous frame to current frame.
        r : (3,) or (3, 1) ndarray
            Estimated translation vector (in 3D).
        f_r_prev_inliers : (M, 2) ndarray
            The inlier subset of right-image points from the "previous" frame.
        f_r_cur_inliers : (M, 2) ndarray
            The inlier subset of right-image points from the "current" frame.
        """

        # Unpack the camera parameters for stereo triangulation
        B   = self.cam.baseline        # Baseline (distance between left & right cameras)
        f   = self.cam.f_len           # Focal length (often fx == fy)
        cu  = self.cam.cu              # Principal point X
        cv  = self.cam.cv              # Principal point Y

        #------------------------------------------------------------
        # 1) Triangulate each row into 3D points in the previous frame & current frame
        #    using stereo geometry:  Z = fB / disparity, X = (x - cu)*Z/f, Y = (y - cv)*Z/f
        #------------------------------------------------------------
        def stereo_triangulate(x_left, y_left, x_right, y_right):
            """ Triangulate a single matched pair from rectified stereo images. 
                Returns (X, Y, Z). """
            disparity = (x_left - x_right)
            Z = (f * B) / (disparity)             # depth
            X = (x_left - cu) * Z / f
            Y = (y_left - cv) * Z / f
            return np.array([X, Y, Z], dtype=np.float32)

        pts_prev_3D = []
        pts_cur_3D  = []
        for row in features_coor:
            # row = [prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y]
            xLP, yLP, xRP, yRP = row[0], row[1], row[2], row[3]   # previous frame L/R
            xLC, yLC, xRC, yRC = row[4], row[5], row[6], row[7]   # current  frame L/R

            # Triangulate in both frames
            p_a = stereo_triangulate(xLP, yLP, xRP, yRP)  # 3D point in "previous" camera coords
            p_b = stereo_triangulate(xLC, yLC, xRC, yRC)  # 3D point in "current" camera coords

            pts_prev_3D.append(p_a)
            pts_cur_3D.append(p_b)

        pts_prev_3D = np.asarray(pts_prev_3D)  # shape (N, 3)
        pts_cur_3D  = np.asarray(pts_cur_3D)   # shape (N, 3)

        #------------------------------------------------------------
        # 2) RANSAC to find the best rigid transform that maps pts_prev_3D -> pts_cur_3D
        #    We'll repeatedly sample 3 correspondences, estimate a transform, and count inliers.
        #------------------------------------------------------------
        max_iterations = 2000       
        inlier_tolerance = 0.3     # distance threshold (meters) to consider an inlier
        best_inlier_count = 0
        best_C = np.eye(3)
        best_r = np.zeros((3,1))
        best_inlier_idx = np.arange(len(pts_prev_3D))  

        N = pts_prev_3D.shape[0]
        if N < 3:
            # Not enough points to do RANSAC; return identity
            return np.eye(3), np.zeros(3), features_coor[:,2:4], features_coor[:,6:8]

        # Helper to compute a 3D alignment from 3 or more pairs (SVD).
        def estimate_transform_svd(src_points, dst_points):
            """
            src_points: (k,3)
            dst_points: (k,3)
            Returns rotation C (3,3) and translation r (3,)
            """
            # 1. Centroid
            p_a = np.mean(src_points, axis=0)  # shape (3,)
            p_b = np.mean(dst_points, axis=0)  # shape (3,)

            # 2. W matrix
            #    Build covariance-like matrix sum( (pb_j - p_b)*(pa_j - p_a)^T ).
            W = np.zeros((3, 3))
            for s, d in zip(src_points, dst_points):
                W += np.outer((d - p_b), (s - p_a))

            # 3. SVD
            U, S, Vt = np.linalg.svd(W)

            V = Vt.T
            # Proper rotation check
            R_ = np.dot(U, V.T)
            if np.linalg.det(R_) < 0:
                # Fix reflection if it occurs
                # Flip the last column of U or V
                U[:, -1] *= -1
                R_ = np.dot(U, V.T)

            # 4. Compute translation
            t_ = p_b - R_.dot(p_a)

            return R_, t_

        rng = np.random.default_rng(seed=1217) 
        for _ in range(max_iterations):
            # Randomly pick 3 distinct indices
            sample_idx = rng.choice(N, size=3, replace=False)
            sample_prev = pts_prev_3D[sample_idx]  # shape (3,3)
            sample_cur  = pts_cur_3D[sample_idx]

            # Estimate transform from these 3 points
            try:
                C_candidate, r_candidate = estimate_transform_svd(sample_prev, sample_cur)
            except np.linalg.LinAlgError:
                # Degenerate sample, skip
                continue

            # Apply transform to all points from the "previous" frame
            pts_prev_transformed = (C_candidate @ pts_prev_3D.T).T + r_candidate

            # Measure distances to the corresponding "current" points
            residuals = np.linalg.norm(pts_cur_3D - pts_prev_transformed, axis=1)

            # Count inliers
            inlier_mask = (residuals < inlier_tolerance)
            num_inliers = np.sum(inlier_mask)

            # Keep track of the best solution
            if num_inliers > best_inlier_count:
                best_inlier_count = num_inliers
                best_C = C_candidate
                best_r = r_candidate.reshape(3,1)
                best_inlier_idx = np.where(inlier_mask)[0]

        #------------------------------------------------------------
        # 3) Re-run the transform estimation on the best inlier set (Weighted SVD).
        #    Typically, we just treat each inlier with equal weight = 1.
        #------------------------------------------------------------
        inlier_pts_prev = pts_prev_3D[best_inlier_idx]
        inlier_pts_cur  = pts_cur_3D[best_inlier_idx]
        C_final, r_final = estimate_transform_svd(inlier_pts_prev, inlier_pts_cur)

        #------------------------------------------------------------
        # 4) Filter the corresponding "right image" features (for display)
        #    We only keep inliers in the right-image coords.
        #------------------------------------------------------------
        # features_coor columns: prev_l_x, prev_l_y, prev_r_x, prev_r_y,
        #                        cur_l_x,  cur_l_y,  cur_r_x, cur_r_y
        # best_inlier_idx tells us which rows are inliers.
        f_r_prev_inliers = features_coor[best_inlier_idx, 2:4]
        f_r_cur_inliers  = features_coor[best_inlier_idx, 6:8]

        # Convert final r to shape (3,)
        r_vec = r_final.reshape(3,)

        # Return the final results
        return C_final, r_vec, f_r_prev_inliers, f_r_cur_inliers

    
    def processFirstFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        
        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right
    
    def processSecondFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
    
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6],img_left, color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right, color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME
        
        return img_l_tracking, img_r_tracking

    def processFrame(self, img_left, img_right, frame_id):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)

        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right,  color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking
    
    def update(self, img_left, img_right, frame_id):
               
        self.new_frame_left = img_left
        self.new_frame_right = img_right
        
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            frame_left, frame_right = self.processFrame(img_left, img_right, frame_id)
            
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            frame_left, frame_right = self.processSecondFrame(img_left, img_right)
            
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            frame_left, frame_right = self.processFirstFrame(img_left, img_right)
            
        self.last_frame_left = self.new_frame_left
        self.last_frame_right= self.new_frame_right
        
        return frame_left, frame_right 


