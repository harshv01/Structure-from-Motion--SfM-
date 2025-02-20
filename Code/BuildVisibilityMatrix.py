import numpy as np

def BuildVisibilityMatrix(X_all_flag, all_feature_flags, cam_num):
    idx = np.where(X_all_flag == 1)[0]
    V = all_feature_flags[idx, :cam_num+1]
    # V = np.transpose(V)

    return idx, V