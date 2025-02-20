import numpy as np

def EstimateFundamentalMatrix(image1_points, image2_points):
    """
    Estimates a fundamental matrix F such that
        image2_points.T @ F @ image1_points = 0.

    Args:
    - image1_points: coordinates of points in image 1
    - image2_points: corresponding coordinates of points in image 2

    Returns:
    - F: returns Fundamental matrix F of shape (3,3)
    """
    A = np.zeros((len(image1_points), 9))
    for i in range(len(image1_points)):

        x1, y1 = image1_points[i]
        x2, y2 = image2_points[i]
        A[i] = [x1 * x2, x2 * y1, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]

    # solve for Ax = 0
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1, :].reshape(3, 3)
    U1, S1, V1t = np.linalg.svd(F)
    S1 = np.diag(S1)
    S1[2, 2] = 0
    F = U1 @ S1 @ V1t

    return F