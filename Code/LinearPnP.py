import numpy as np

from Utils.Utils import get_homogeneous

def pnp_reproj_err(X, x, K, R, C):
    P = K @ R @ np.concatenate((np.eye(3), -1 * C.reshape((-1,1))), axis=1)
    
    error = []
    for X, pt in zip(X, x):

        P1T, P2T, P3T = P
        P1T, P2T, P3T = P1T.reshape(1,-1), P2T.reshape(1,-1), P3T.reshape(1,-1)
        X = np.hstack((X.reshape(1,-1), np.ones((pts.shape[0], 1)))).reshape(-1,1)
        u, v = pt[0], pt[1]
        u_proj = np.divide(P1T.dot(X) , P3T.dot(X))
        v_proj =  np.divide(P2T.dot(X) , P3T.dot(X))

        e = (v - v_proj)**2 + (u - u_proj)**2

        error.append(e)
        
    return np.mean(np.array(error).squeeze())

def linear_pnp(projected_world_points, img_points, K):
    A = []
    for i in range(len(projected_world_points)):
        X, Y, Z = projected_world_points[i]
        x, y = img_points[i]

        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])

    A = np.array(A)

    U, W, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    gamma_R = np.linalg.inv(K) @ P[:, :3]
    U, W, Vt = np.linalg.svd(gamma_R)
    R = U @ Vt

    T = (np.linalg.inv(K) @ P[:, 3]) / W[0]
    if np.linalg.det(R) < 0:
        R = -1 * R
        T = -T
        
    C = -R.T @ T
    return R, C