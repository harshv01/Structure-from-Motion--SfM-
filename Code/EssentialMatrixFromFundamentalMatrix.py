import numpy as np

def getEssentialMatrix(K, F):
    E = np.transpose(K) @ F @ K
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    E = U @ W @ Vt
    return E