import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from Utils.Utils import *
from BuildVisibilityMatrix import BuildVisibilityMatrix
import matplotlib.pyplot as plt

I_ID = 0

def get_2D_points(X_index, visiblity_matrix, feature_x, feature_y):
    pts2D = []
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = np.hstack((visible_feature_x[i,j], visible_feature_y[i,j]))
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2)             

def get_cam_and_point_indices(visiblity_matrix):
    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)

def bundle_adjustment_sparsity(X_flag, best_x_flag, num_cam):
    """
    https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    """
    n_cameras = num_cam + 1
    X_index, visiblity_matrix = BuildVisibilityMatrix(X_flag.reshape(-1), best_x_flag, num_cam)
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_index)

    m = int(n_observations * 2)
    n = int(n_cameras * 6 + n_points * 3)
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(n_observations)
    camera_indices, point_indices = get_cam_and_point_indices(visiblity_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (num_cam)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (num_cam) * 6 + point_indices * 3 + s] = 1

    return A

def project_point(R, C, point, K):
    P = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3, 1)))))
    homo_point = np.hstack((point, 1)).reshape(-1, 1)
    point_proj = np.dot(P, homo_point)
    point_proj /= point_proj[-1]
    return point_proj.ravel()

def project(points, camera_params, K):
    points_proj = []
    for i in range(len(camera_params)):
        R = Rotation.from_rotvec(camera_params[i, :3])
        R = R.as_matrix()
        C = camera_params[i, 3:].reshape(3,1)
        point = points[i]
        point_proj = project_point(R, C, point, K)[:2]
        points_proj.append(point_proj)
    return np.array(points_proj)

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def fun(x0, num_cam, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    """
    number_of_cam = num_cam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    
    return error_vec

def BundleAdjustment(X_all, X_flag, xx_coord, xy_coord, best_feature_flag, R_set, C_set, K, num_cam):
    """
    Bundle Adjustment using least squares with sparsity enabled to reduce residue

    Args:
        X_all: All Triangulated points
        X_flag: Flag of Triangulated points
        xx_coord: x coordinate of x
        xy_coord: y coordinate of y
        best_feature_flag: feature flags after RANSAC
        R_set: Rotation Matrix
        C_set: Camera Center
        K: Camera Intrinsic Matrix
        num_cam : num of current camera - 1

    Returns:
        R,C,X
    """
    
    X_index, visiblity_matrix = BuildVisibilityMatrix(X_flag, best_feature_flag, num_cam)
    X = X_all[X_index]
    x = get_2D_points(X_index, visiblity_matrix, xx_coord, xy_coord)

    n_points = X.shape[0]

    x0 = []
    for i in range(num_cam+1):
        C, R = C_set[i], R_set[i]
        euler = Rotation.from_matrix(R)
        Q = euler.as_rotvec()
        try:
            RC = [Q[0], Q[1], Q[2], C[0][0], C[1][0], C[2][0]]
        except:
            RC = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        x0.extend(RC)

    x0.extend(list(X.flatten()))
    x0 = np.array(x0)

    camera_indices, point_indices = get_cam_and_point_indices(visiblity_matrix)
    
    global I_ID
    
    f0 = fun(x0, num_cam, n_points, camera_indices, point_indices, x, K)
    plt.plot(f0)
    plt.savefig(f'initResidual{I_ID}.png')    
    
    A = bundle_adjustment_sparsity(X_flag, best_feature_flag, num_cam)
    optim = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                        args=(num_cam, n_points, camera_indices, point_indices, x, K))
    
    x1 = optim.x
    
    plt.plot(optim.fun)
    plt.savefig(f'FinalResidual{I_ID}.png')
    plt.clf()
    I_ID += 1
    
    n_cameras = num_cam + 1
    optim_camera_params = x1[:n_cameras * 6].reshape((n_cameras, 6))
    optim_X = x1[n_cameras * 6:].reshape((n_points, 3))

    optim_X_all = np.zeros_like(X_all)
    optim_X_all[X_index] = optim_X

    optim_C_set, optim_R_set = [], []
    for i in range(len(optim_camera_params)):
        R = Rotation.from_rotvec(optim_camera_params[i,:3])
        R = R.as_matrix()
        C = optim_camera_params[i, 3:].reshape(3,1)
        optim_C_set.append(C)
        optim_R_set.append(R)
    
    return optim_R_set, optim_C_set, optim_X_all