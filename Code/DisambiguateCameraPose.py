import numpy as np

def CheckCheirality(X, C, R):
    """
    Check cheirality condition for triangulated points.

    Args:
    - X: triangulated 3D points
    - C: Camera center
    - R: Rotation matrix

    Returns:
    - n: number of points were point is in front of camera
    """
    N = len(X)
    n = 0
    mask = np.zeros(N, dtype=bool)
    for i in range(N):
        x = X[i][:3].reshape(-1,1)
        if (R[2, :] @ (x - C) > 0) and x[2] > 0:
            n += 1
    return n

def DisambiguatePose(X, R_set, C_set):
    best_i = 0
    best_n = 0
    
    for i in range(len(R_set)):
        n = CheckCheirality(X[i], C_set[i].reshape(-1,1), R_set[i])
        if n > best_n:
            best_i = i
            best_n = n

    return R_set[best_i], C_set[best_i], X[best_i]