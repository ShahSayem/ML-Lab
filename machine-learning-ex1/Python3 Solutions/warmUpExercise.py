import numpy as np

def warm_up_exercise(): # The function warm_up_exercise returns a 5x5 identity matrix.
    return np.eye(5) # The identity matrix is a square matrix with ones on the diagonal and zeros elsewhere.


# Example usage:
if __name__ == "__main__":
    identity_matrix = warm_up_exercise()
    print("5x5 Identity Matrix:\n", identity_matrix)
