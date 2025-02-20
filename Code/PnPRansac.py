from LinearPnP import linear_pnp
import numpy as np
from Utils.Utils import get_homogeneous


def pnp_reprojection_error(x, X, R, C, K):
    u,v = x
    X = get_homogeneous(X.reshape(1,-1)).reshape(-1,1)
    C = C.reshape(-1, 1)
    P = K @ R @ np.concatenate((np.eye(3), -1 * C), axis=1)
    p1, p2, p3 = P
        
    u_proj = (p1 @ X) / (p3 @ X)
    v_proj = (p2 @ X) / (p3 @ X)

    x_proj = np.hstack((u_proj, v_proj))
    x = np.hstack((u, v))
    return np.linalg.norm(x - x_proj)
    

def PnPRANSAC(K, x, X, n_max=1000, threshold=5):
    best_R, best_C = None, None
    n_rows = X.shape[0]
    best_n = -1
    
    for _ in range(n_max):
        
        random_indices = np.random.choice(n_rows, size=6)
        X_set, x_set = X[random_indices], x[random_indices]
        
        R, C = linear_pnp(X_set, x_set, K)
        
        indices = []
        if R is not None:
            for j in range(n_rows):
                feature = x[j]
                Xj = X[j]
                error = pnp_reprojection_error(feature, Xj, R, C, K)

                if error < threshold:
                    indices.append(j)
                    
        if len(indices) > best_n:
            best_n = len(indices)
            best_R = R
            best_C = C
            

    return best_R, best_C
