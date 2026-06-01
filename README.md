# AER1217 — Development of Autonomous UAS

**University of Toronto (UTIAS) | MEng Aerospace Science & Engineering | Winter/Spring 2025**

A graduate course spanning the full autonomy stack for unmanned aerial systems: dynamics and control, computer vision, state estimation, and real-time path planning — implemented on both real hardware (Crazyflie, Parrot AR.Drone) and PyBullet simulation.

---

## Labs & Project

| Folder | Topic | Key Methods |
|--------|-------|-------------|
| [Lab 3](Deliverables/Lab_3/) | Object detection + geo-localization | OpenCV HSV detection, back-projection, K-means + RANSAC |
| [Lab 4](Deliverables/Lab_4/) | Stereo Visual Odometry | SIFT features, epipolar filtering, RANSAC + SVD pose estimation on KITTI |
| [Project](Deliverables/Project/) | Autonomous quadrotor path planning | 3D grid search, obstacle avoidance — **1st place in class competition** |

---

## Skills Demonstrated

- **Computer vision:** SIFT feature detection, stereo epipolar geometry, HSV color segmentation
- **State estimation:** RANSAC outlier rejection, SVD rigid-body alignment, back-projection
- **Path planning:** 3D trajectory generation with obstacle avoidance in PyBullet
- **Real drone integration:** Parrot AR.Drone pose-fused perception pipeline

## Tools

Python · NumPy · OpenCV · PyBullet · Crazyflie SDK · SciPy · Matplotlib
