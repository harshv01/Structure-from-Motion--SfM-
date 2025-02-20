import numpy as np
from EstimateFundamentalMatrix import *

def getInliers(x1, x2, indices, n_max=2000, threshold=0.05):
    best_indices = []
    best_F = None

    for i in range(n_max):
  
        #select 8 points randomly
        n_rows = x1.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        pts1_8 = x1[random_indices, :] 
        pts2_8 = x2[random_indices, :] 
        F = EstimateFundamentalMatrix(pts1_8, pts2_8)
        indices_i = []
        if F is not None:
            for j in range(n_rows):
                x1j, x2j = x1[j, :], x2[j, :]
                x1j=np.array([x1j[0], x1j[1], 1])
                x2j=np.array([x2j[0], x2j[1], 1])
                error = np.abs(np.dot(x2j.T, np.dot(F, x1j)))
                
                if error < threshold:
                    indices_i.append(indices[j])

        if len(indices_i) > len(best_indices):
            best_indices = indices
            best_F = F

    return best_F, best_indices



