import numpy as np

def ExtractCameraPose(E):
    """
    return set of Rotation and Camera Centers
    """

    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R, C = [], []
    R.append(U @ (W @ Vt))
    R.append(U @ (W @ Vt))
    R.append(U @ (W.T @ Vt))
    R.append(U @ (W.T @ Vt))
    C.append(U[:, 2].reshape((3,1)))
    C.append(-U[:, 2].reshape((3,1)))
    C.append(U[:, 2].reshape((3,1)))
    C.append(-U[:, 2].reshape((3,1)))

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return np.array(R), np.array(C)

