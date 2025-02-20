# Structure From Motion with Traditional Approach

## Introduction
SfM is used to create a 3D scene from a given set of images. We reconstruct a 3D scene and simultaneously obtain the camera poses of a monocular camera w.r.t. the given scene. Here, we compiled 5 images of Unity Hall at Worcester Polytechnic Institute to obtain a reconstructed scene of the building.

<img src="https://github.com/user-attachments/assets/ea12167e-6071-4f89-8fe4-4ed8edf0906d" height="150">

The Traditional approach follows a series of steps to achieve this, which are outlined below.

## Steps
1. **Feature Matching and Outlier rejection using RANSAC**
2. **Estimate Fundamental Matrix**
3. **Estimating Essential Matrix from Fundamental Matrix**
4. **Estimate Camera Pose from Essential Matrix**
5. **Cheirality Check using Triangulation**
6. **Perspective-n-Point (PnP)**
7. **PnP RANSAC**
8. **Bundle Adjustment**

## Results
The repository includes the implementation and results of each step in the SfM process. The reprojected errors after each step are provided in a table for analysis. <br>
<img src="https://github.com/user-attachments/assets/17a28cb7-23e8-43de-b810-9cfb0e84719b" height="400">
<img src="https://github.com/user-attachments/assets/13763030-07b9-4eb2-8f22-d4691794eac3" height="400">
<img width="643" alt="result_table" src="https://github.com/user-attachments/assets/8a78aa3f-bac7-45d1-ab6d-2c1c1d80cbb5" />

## Usage
To run the code and reproduce the results, follow these steps:
1. Clone the repository.
2. Navigate to the directory containing the code.
3. Run the Wrapper File to execute the SfM pipeline.

## Dependencies
The code in this repository requires the following dependencies:
- Python 3.x
- NumPy
- OpenCV
- SciPy
