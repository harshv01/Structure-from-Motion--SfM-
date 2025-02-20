
import scipy.optimize as optimize
import numpy as np
from Utils.Utils import *


def NonLinearPnP(K, pts, X, R0, C0):

    Q = getQuaternion(R0)
    X0 = [Q[0] ,Q[1],Q[2],Q[3], C0[0], C0[1], C0[2]] 

    optimized_params = optimize.least_squares(
        fun = reproj_err,
        x0=X0,
        method="trf",
        args=[X, pts, K])
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = Rotation.from_quat(Q)
    R = R.as_matrix()
    return R, C

def reproj_err(X0, world_points, img_pts, K):
    
    Q, C = X0[:4], X0[4:].reshape(-1,1)
    R = Rotation.from_quat(Q)
    R.as_matrix()
    P = K @ R @ np.concatenate((np.eye(3), -1 * C.reshape((-1,1))), axis=1)
    
    error = []
    for X, img_pt in zip(world_points, img_pts):

        P1T, P2T, P3T = P
        P1T, P2T, P3T = P1T.reshape(1,-1), P2T.reshape(1,-1), P3T.reshape(1,-1)


        X = get_homogeneous(X.reshape(1,-1)).reshape(-1,1)
        u, v = img_pt[0], img_pt[1]
        u_proj = np.divide(P1T.dot(X) , P3T.dot(X))
        v_proj =  np.divide(P2T.dot(X) , P3T.dot(X))

        e = (u - u_proj)**2 + (v - v_proj)**2

        error.append(e)

        return np.mean(np.array(error).squeeze())