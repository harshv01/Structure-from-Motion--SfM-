import numpy as np

def skew_matrix(x):
    X = np.array([[0, -1 , x[1]],
                  [1, 0, -x[0]],
                  [-x[1], x[0], 0]])
    return X

def Triangulate(x1, x2, P1, P2):
    """
    Triangulates 3D points given point correspondences and camera projection matrices.

    Args:
    - x1: Coordinates of points in image 1
    - x2: Corresponding coordinates of points in image 2
    - P1: Camera projection matrix of image 1 with shape (3, 4)
    - P2: Camera projection matrix of image 1 with shape (3, 4)

    Returns:
    - X: returns the triangulated 3D points with shape (N, 4) in homogeneous
    """
    N = len(x1)
    X = np.zeros((N, 4))
    for i in range(N):

        X1_i = skew_matrix(x1[i]) @ P1
        X2_i = skew_matrix(x2[i]) @ P2
        x_P = np.vstack((X1_i, X2_i))

        _, _, Vt = np.linalg.svd(x_P)
        X_non_homogeneous = Vt[-1]
        # convert (ax, ay, az, a) to (x, y, z, 1)
        X_homogeneous = X_non_homogeneous / X_non_homogeneous[-1]
        X[i] = X_homogeneous
    return X