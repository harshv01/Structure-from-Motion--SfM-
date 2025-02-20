import numpy as np
import scipy.optimize as optimize

def NonLinearTriangulation(K, x1, x2, X, R1, C1, R2, C2):
    
    P1 = K @ R1 @ np.concatenate((np.eye(3), -1 * C1.reshape((-1,1))), axis=1)
    P2 = K @ R2 @ np.concatenate((np.eye(3), -1 * C2.reshape((-1,1))), axis=1)
    
    X_optim = []
    for i in range(len(X)):
        optimized_params = optimize.least_squares(fun=reprojection_error, x0=X[i], method="trf", args=[x1[i], x2[i], P1, P2])
        X1 = optimized_params.x
        X_optim.append(X1)
    return np.array(X_optim)


def reprojection_error(X, x1, x2, P1, P2):
    u1, v1 = x1
    u2, v2 = x2
    error_1 = (u1 - (P1[0,:] @ X) / (P1[2,:] @ X)) ** 2 + (v1 - (P1[1,:] @ X) / (P1[2,:] @ X)) ** 2
    error_2 = (u2 - (P2[0,:] @ X) / (P2[2,:] @ X)) ** 2 + (v2 - (P2[1,:] @ X) / (P2[2,:] @ X)) ** 2
    error = error_1 + error_2

    return error



