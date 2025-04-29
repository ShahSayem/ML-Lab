import numpy as np
from computeCostMulti import compute_cost_multi

def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for _ in range(num_iters):
        error = X @ theta - y
        theta -= (alpha / m) * (X.T @ error)
        J_history.append(compute_cost_multi(X, y, theta))
    return theta, J_history