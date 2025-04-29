import numpy as np

def compute_cost_multi(X, y, theta):
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)


# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2, 3])
    theta = np.array([0.5, 0.5])

    cost = compute_cost_multi(X, y, theta)
    print("Computed cost:", cost)
# The above code defines a function to compute the cost for multiple linear regression.