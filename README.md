# Structure From Motion with Traditional Approach

## Introduction
This repository contains the implementation of Structure from Motion (SfM). SfM is used to create a 3D scene from a given set of images. The Traditional approach follows a series of steps to achieve this, which are outlined below.

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
The repository includes the implementation and results of each step in the SfM process. The reprojected errors after each step are provided in a table for analysis.
![Table Results](Phase1/result_table.png)

![2D representation of the Projected Points](Phase1/Results/2D.png)
![3D representation of the Projected Points](Phase1/Results/3D.png)

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