import numpy as np

def normal_eqn(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y