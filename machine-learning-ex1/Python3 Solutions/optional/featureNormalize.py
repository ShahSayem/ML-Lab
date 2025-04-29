import numpy as np

def feature_normalize(X):
    mu = np.mean(X, axis=0)       # Mean for each feature/column
    sigma = np.std(X, axis=0)     # Standard deviation for each feature/column
    X_norm = (X - mu) / sigma     # Feature-wise normalization
    return X_norm, mu, sigma



# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [3, 4], [5, 6]])

    X_norm, mu, sigma = feature_normalize(X)
    print("Normalized features:\n", X_norm)
    print("Mean:\n", mu)
    print("Standard deviation:\n", sigma)
# The above code defines a function to normalize features in a dataset.
# It standardizes the features by removing the mean and scaling to unit variance.