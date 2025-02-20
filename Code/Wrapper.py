import numpy as np

from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from NonLinearTriangulation import *
from DisambiguateCameraPose  import *
from LinearPnP import *
from PnPRansac import *
from NonLinearPnP import *
from BundleAdjustment import *
from Utils.Utils import *

from matplotlib import pyplot as plt

np.random.seed(42)

K = read_calib_matrix()
total_images = 5

def main():
    
    all_features_flag, all_features_x, all_features_y = get_feature_and_flags(num_cams=5, file_path='../Data/matching')
    
    best_features_flag = np.zeros_like(all_features_flag)

    F = None

    print("Loading all images in a list")
    all_images = read_images()   # using default argument values

    ## -----features before any RANSAC
    x1_plot = np.concatenate((all_features_x[:, 0].reshape((-1,1)), all_features_y[:, 0].reshape((-1,1))), axis=1)
    x2_plot = np.concatenate((all_features_x[:, 1].reshape((-1,1)), all_features_y[:, 1].reshape((-1,1))), axis=1)
    plot_matches(all_images[0], all_images[1], len(x1_plot), x1_plot, x2_plot, 0, "1_before")   # Ransac result display

    print("Computing Inliners RANSAC for every pair of images")
    for i in range(total_images - 1):
        for j in range(i + 1, total_images):
            idx = np.where((all_features_flag[:,i]==1) & (all_features_flag[:,j]==1))[0]
            x1 = np.concatenate((all_features_x[idx, i].reshape((-1,1)), all_features_y[idx, i].reshape((-1,1))), axis=1)
            x2 = np.concatenate((all_features_x[idx, j].reshape((-1,1)), all_features_y[idx, j].reshape((-1,1))), axis=1)

            idx = np.array(idx)

            if len(idx) > 8:
                F_best, chosen_idx = getInliers(x1, x2, idx, n_max=1000)                
                
                if i==0 and j==1:
                    ## -----features after 8 pointer RANSAC
                    x1_plot = np.concatenate((all_features_x[chosen_idx, i].reshape((-1,1)), all_features_y[chosen_idx, i].reshape((-1,1))), axis=1)
                    x2_plot = np.concatenate((all_features_x[chosen_idx, j].reshape((-1,1)), all_features_y[chosen_idx, j].reshape((-1,1))), axis=1)
                    plot_matches(all_images[i], all_images[j], len(chosen_idx), x1_plot, x2_plot, i, "1_after")   # Ransac result display
                
                    F = F_best

                best_features_flag[chosen_idx, j] = 1
                best_features_flag[chosen_idx, i] = 1
            else:
                print("Insufficient number of points for 8 point algorithm")
                exit()
                
    E = getEssentialMatrix(K, F)
    R_set, C_set = ExtractCameraPose(E)

    # get points for image 0 and image 1
    idx = np.where((best_features_flag[:,0]==1) & (best_features_flag[:,1]==1))[0]
    x1 = np.concatenate((all_features_x[idx, 0].reshape((-1,1)), all_features_y[idx, 0].reshape((-1,1))), axis=1)
    x2 = np.concatenate((all_features_x[idx, 1].reshape((-1,1)), all_features_y[idx, 1].reshape((-1,1))), axis=1)

    R0 = np.identity(3)
    C0 = np.zeros((3,1))
    
    print("Computing Triangulation")
    triangulated_x = []
    for i in range(len(C_set)):
        P1 = K @ R0 @ np.concatenate((np.eye(3), -1 * C0), axis=1)
        P2 = K @ R_set[i] @ np.concatenate((np.eye(3), -1 * C_set[i]), axis=1)
        X = Triangulate(x1, x2, P1, P2)
        triangulated_x.append(X)

    triangulated_x = np.array(triangulated_x)
    best_R, best_C, X = DisambiguatePose(triangulated_x, R_set, C_set)
    
    X = X/X[:,3].reshape(-1,1)
    
    X_optim = NonLinearTriangulation(K, x1, x2, X, R0, C0, best_R, best_C)
    X_optim = X_optim / X_optim[:,3].reshape(-1,1)
    
    # # show reproj points, before and after
    plot_reprojection(x1, X, all_images[0], K, best_C, best_R, 'reprojections_before')
    plot_reprojection(x1, X_optim, all_images[0], K, best_C, best_R, 'reprojections_after')

    mean_error1 = avg_reproj_err(X, x1, x2, R0, C0, best_R, best_C, K )
    mean_error2 = avg_reproj_err(X_optim, x1, x2, R0, C0, best_R, best_C, K )
    print(1,2, 'Before optimization LT: ', mean_error1, 'After optimization nLT:', mean_error2)

    X_all = np.zeros((all_features_x.shape[0], 3))
    X_flags = np.zeros((all_features_x.shape[0], 1), dtype = int)

    X_all[idx] = X[:, :3]
    X_flags[idx] = 1
    X_flags[np.where(X_all[:,2] < 0)] = 0

    C_set, R_set = [], []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set.append(C0)
    R_set.append(R0)

    C_set.append(best_C)
    R_set.append(best_R)
    
    for i in range(2, total_images):

        print(f'Adding image: {i+1}')
        idx_i = np.where((X_flags[:, 0]==1) & (best_features_flag[:, i]==1))[0]

        xi = np.hstack((all_features_x[idx_i, i].reshape(-1,1), all_features_y[idx_i, i].reshape(-1,1)))
        X = X_all[idx_i, :].reshape(-1,3)
        
        R, C = PnPRANSAC(K, xi, X, n_max = 1000, threshold = 5)
        errorLinearPnP = pnp_reproj_err(X, xi, K, R, C)
        
        Ri, Ci = NonLinearPnP(K, xi, X, R, C)
        errorNonLinearPnP = pnp_reproj_err(X, xi, K, Ri, Ci)
        print("Error after linear PnP: ", errorLinearPnP, " Error after non linear PnP: ", errorNonLinearPnP)

        C_set.append(Ci)
        R_set.append(Ri)

        #trianglulation
        for j in range(i):
            match_pts_idx = np.where((best_features_flag[:, j]==1) & (best_features_flag[:, i]==1))[0]

            x1 = np.hstack((all_features_x[match_pts_idx, j].reshape((-1, 1)), all_features_y[match_pts_idx, j].reshape((-1, 1))))
            x2 = np.hstack((all_features_x[match_pts_idx, i].reshape((-1, 1)), all_features_y[match_pts_idx, i].reshape((-1, 1))))

            P1 = K @ R_set[j] @ np.concatenate((np.eye(3), -1 * C_set[j].reshape((-1,1))), axis=1)
            P2 = K @ Ri @ np.concatenate((np.eye(3), -1 * Ci.reshape((-1,1))), axis=1)
            X = Triangulate(x1, x2, P1, P2)
            X = X/X[:,3].reshape(-1,1)
            
            LT_error = avg_reproj_err(X, x1, x2, R_set[j], C_set[j], Ri, Ci, K)
            
            X = NonLinearTriangulation(K, x1, x2, X, R_set[j], C_set[j], Ri, Ci)
            X = X/X[:,3].reshape(-1,1)
            
            nLT_error = avg_reproj_err(X, x1, x2, R_set[j], C_set[j], Ri, Ci, K)
            print("Error after linear triangulation: ", LT_error, " Error after non linear triangulation: ", nLT_error)
            
            X_all[match_pts_idx] = X[:,:3]
            X_flags[match_pts_idx] = 1
            
        print( 'Performing Bundle Adjustment  for image : ', i  )
        R_set, C_set, X_all = BundleAdjustment(X_all,X_flags, all_features_x, all_features_y,
                                                    best_features_flag, R_set, C_set, K, num_cam = i)
        
    
        for k in range(0, i+1):
            match_pts_idx = np.where((X_flags[:,0]==1) & (best_features_flag[:, k]==1))[0]
            x = np.hstack((all_features_x[match_pts_idx, k].reshape((-1, 1)), all_features_y[match_pts_idx, k].reshape((-1, 1))))
            X = X_all[match_pts_idx]
            BA_error = pnp_reproj_err(X, x, K, R_set[k], C_set[k])
            print("Error after BA :", BA_error)
        
    X_flags[X_all[:,2]<0] = 0    
    
    feature_idx = np.where(X_flags[:, 0])
    X = X_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    
    # 2D plotting
    fig = plt.figure(figsize = (10, 10))
    plt.xlim(-100, 50)
    plt.ylim(-10, 250)
    plt.scatter(x, z, marker='.',linewidths=0.5, color = 'blue')
    for i in range(0, len(C_set)):
        euler = Rotation.from_matrix(R_set[i])
        R1 = euler.as_rotvec()
        R1 = np.rad2deg(R1)
        plt.plot(C_set[i][0],C_set[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')
        
    plt.savefig('2D.png')
    plt.show()
    
    # For 3D plotting
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(x, y, z, color = "green")
    ax.set_xlim(-100, 50)
    # ax.set_ylim(y_min, y_max)
    ax.set_zlim(-10, 250)
    plt.savefig('3D.png')
    plt.show()
    
if __name__ == '__main__':
    main()
         

        
