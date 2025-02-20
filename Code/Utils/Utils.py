import numpy as np
import cv2
from scipy.spatial.transform import Rotation 

def read_calib_matrix(file_path='../Data/calibration.txt'):
    """
    reads and returns the Camera Calibration Matrix / Intrinsic Matrix

    Args:
        file_path (str, optional): _description_. Defaults to '../Data/calibration.txt'.

    Returns:
        K: Camera Calibration Matrix / Intrinsic Matrix
    """

    K = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = []
            row_str = line.strip().split()
            for val in row_str:
                row.append(eval(val))
            K.append(row)

    np.set_printoptions(precision=30, suppress=True)
    K = np.array(K, dtype=float)
    return K

def get_feature_and_flags(num_cams=5, file_path='../Data/matching'):
    """
    returns: flags for features, x co-ordinate values, y co-ordinate values for each image
    """

    all_feature_flags = []
    all_features_x = []
    all_features_y = []


    for i in range(0, num_cams - 1):

        with open(file_path + str(i + 1) + '.txt', 'r') as file:
            lines = file.readlines()
            latest_index = 0
            for j in range(1, len(lines)):
                line = lines[j]
                split_line = line.strip().split()
                feature_flag_row = np.zeros(num_cams, )
                feature_flag_row[i] = 1

                u_self, v_self = eval(split_line[4]), eval(split_line[5])

                features_row_x = np.zeros(num_cams, )
                features_row_y = np.zeros(num_cams, )
                features_row_x[i] = u_self
                features_row_y[i] = v_self

                for k in range(6, len(split_line), 3):
                    feature_flag_row[eval(split_line[k]) - 1] = 1
                    match_u, match_v = eval(split_line[k + 1]), eval(split_line[k + 2])
                    features_row_x[eval(split_line[k]) - 1] = match_u
                    features_row_y[eval(split_line[k]) - 1] = match_v

                all_feature_flags.append(feature_flag_row)
                all_features_x.append(features_row_x)
                all_features_y.append(features_row_y)

    all_feature_flags = np.array(all_feature_flags)
    all_features_x = np.array(all_features_x)
    all_features_y = np.array(all_features_y)

    return all_feature_flags, all_features_x, all_features_y
        

def avg_reproj_err(world_points, x1s, x2s, R1, C1, R2, C2, K ):
    error = []
    for x1, x2, X in zip(x1s, x2s, world_points):
        error1,error2 = reproj_err(X, x1, x2, R1, C1, R2, C2, K )
        error.append(error1+error2)
        
    return np.mean(error)

def reproj_err(X, x1, x2, R1, C1, R2, C2, K ):
    P1 = K @ R1 @ np.concatenate((np.eye(3), -1 * C1.reshape((-1,1))), axis=1)
    P2 = K @ R2 @ np.concatenate((np.eye(3), -1 * C2.reshape((-1,1))), axis=1)
    
    u1, v1 = x1
    u2, v2 = x2
    
    error_1 = (u1 - (P1[0,:] @ X) / (P1[2,:] @ X)) ** 2 + (v1 - (P1[1,:] @ X) / (P1[2,:] @ X)) ** 2
    error_2 = (u2 - (P2[0,:] @ X) / (P2[2,:] @ X)) ** 2 + (v2 - (P2[1,:] @ X) / (P2[2,:] @ X)) ** 2
    return error_1, error_2

def plot_matches(img1, img2, num_inliers, pixel_coors1, pixel_coors2, img1_name, img2_name):    # Plots inliers or rejects, whichever is passed
    
    # pixel_coors1 = [i[0] for i in match_pairs]
    # pixel_coors2 = [i[1] for i in match_pairs]
    
    keypoints_1 = conv_coors_key(pixel_coors1, 5)
    keypoints_2 = conv_coors_key(pixel_coors2, 5)

    cv2_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(num_inliers)]

    result_img = cv2.drawMatches(img1, 
                            keypoints_1, 
                            img2, 
                            keypoints_2, 
                            cv2_matches, 
                            outImg = None, 
                            matchesThickness = 1,
                            matchColor=(0, 255, 255), 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                            )

    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f'{img1_name}&{img2_name}.png', result_img_rgb)


def conv_coors_key(pixel_coors, size):    #convert coordinates to cv2 keypoint data format
    keypoints = []
    for pixel in pixel_coors:
        keypoints.append(cv2.KeyPoint(float(pixel[0]), float(pixel[1]), size))
    return keypoints
        
def plot_reprojection(pts1, x3D, img, K, C, R, name):
    all_points = []
    for X in x3D:
        P1 = K @ R @ np.concatenate((np.eye(3), -1 * C.reshape((-1,1))), axis=1)
    
        if len(X) == 3:
            X = np.array([X[0],X[1],X[2],1])

        u1, v1 = (P1[0,:] @ X) / (P1[2,:] @ X), (P1[1,:] @ X) / (P1[2,:] @ X)
        all_points.append((u1, v1))
    
    print('shape of pts1:', pts1.shape)
    print('type of pts1', type(pts1))

    plot_reproj_img = img.copy()  
    for pt in pts1:
        plot_reproj_img = cv2.circle(plot_reproj_img, (int(pt[0]), int(pt[1])), radius=3, color=[0, 0, 255], thickness=-1)  ## Red is old

    for pt in all_points:
        plot_reproj_img = cv2.circle(plot_reproj_img, (int(pt[0]), int(pt[1])), radius=3, color=[0, 255, 0], thickness=-1)  ## Green is new

    cv2.imwrite(f'./{name}.png', plot_reproj_img)

def read_images(path='../Data/', num_cams = 5):
    all_img = []
    for i in range(1, num_cams+1):
        all_img.append(cv2.imread(f'{path}{i}.png'))

    return all_img
