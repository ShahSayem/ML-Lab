import numpy as np
from computeCost import compute_cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for _ in range(num_iters):
        error = X @ theta - y
        theta -= (alpha / m) * (X.T @ error)
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2, 3])
    theta = np.array([0.5, 0.5])
    alpha = 0.01
    num_iters = 1000

    theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
    print("Theta after gradient descent:", theta)
    print("Cost history:", J_history)
# The above code implements gradient descent for linear regression.
# It updates the model parameters (theta) iteratively to minimize the cost function.