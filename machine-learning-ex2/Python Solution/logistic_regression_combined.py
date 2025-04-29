import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression cost function
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    J = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    grad = (1 / m) * (X.T @ (h - y))
    return J, grad

# Logistic regression with regularization
def cost_function_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    J = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + reg
    grad = (1 / m) * (X.T @ (h - y))
    grad[1:] += (lambda_ / m) * theta[1:]
    return J, grad

# Prediction
def predict(theta, X):
    return sigmoid(X @ theta) >= 0.5

# Polynomial feature mapping
def map_feature(X1, X2, degree=6):
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.hstack((out, (X1**(i - j) * X2**j).reshape(-1, 1)))
    return out

# Plotting data
def plot_data(X, y):
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], c='y', edgecolors='k', label='Not admitted')

# Plotting decision boundary
def plot_decision_boundary(theta, X, y, mapFeatureUsed=False):
    plot_data(X[:, 1:3], y)
    if X.shape[1] <= 3:
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y, label='Decision Boundary')
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                mapped = map_feature(np.array([u[i]]), np.array([v[j]]))
                z[i, j] = mapped @ theta
        z = z.T
        plt.contour(u, v, z, levels=[0], linewidths=2)
    plt.xlabel('Exam 1 score' if not mapFeatureUsed else 'Microchip Test 1')
    plt.ylabel('Exam 2 score' if not mapFeatureUsed else 'Microchip Test 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==== EXERCISE 2: Logistic Regression ====

def run_logistic_regression():
    print("==== Logistic Regression without Regularization ====")
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2].reshape(-1, 1)
    plot_data(X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()

    # Add intercept
    m, n = X.shape
    X = np.hstack([np.ones((m, 1)), X])
    initial_theta = np.zeros((n + 1, 1))

    cost, grad = cost_function(initial_theta, X, y)
    print(f"Initial cost: {cost[0]:.4f}")
    print("Initial gradient:", grad.ravel())

    # Optimization
    res = minimize(lambda t: cost_function(t, X, y), initial_theta.ravel(), jac=True, method='TNC')
    theta = res.x.reshape(-1, 1)
    print(f"Optimized cost: {res.fun:.4f}")
    print("Optimized theta:", theta.ravel())

    plot_decision_boundary(theta, X, y)

    # Predict
    prob = sigmoid(np.array([1, 45, 85]) @ theta)
    print(f"Admission probability for scores 45 & 85: {prob[0]:.4f}")
    p = predict(theta, X)
    print(f"Training Accuracy: {np.mean(p == y) * 100:.2f}%")

# ==== EXERCISE 2 (Part 2): Regularized Logistic Regression ====

def run_regularized_logistic_regression():
    print("\\n==== Regularized Logistic Regression ====")
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2].reshape(-1, 1)
    plot_data(X, y)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.show()

    # Map Features
    X_mapped = map_feature(X[:, 0], X[:, 1])
    initial_theta = np.zeros((X_mapped.shape[1], 1))

    # Try different lambda values
    for lambda_ in [0, 1, 10, 100]:
        print(f"\n-- Training with lambda = {lambda_} --")
        cost, grad = cost_function_reg(initial_theta, X_mapped, y, lambda_)
        res = minimize(lambda t: cost_function_reg(t, X_mapped, y, lambda_), initial_theta.ravel(), jac=True, method='TNC')
        theta = res.x.reshape(-1, 1)
        plot_decision_boundary(theta, X_mapped, y, mapFeatureUsed=True)
        plt.title(f"Decision Boundary (lambda = {lambda_})")
        p = predict(theta, X_mapped)
        print(f"Training Accuracy (lambda = {lambda_}): {np.mean(p == y) * 100:.2f}%")

# Run both parts
print("==== Logistic Regression and Regularization ====")
print("Choose one of the following options:")
print("1. Logistic Regression without Regularization\n2. Regularized Logistic Regression")
choice = input("Enter your choice (1 or 2): ")

if choice == '1':
    run_logistic_regression()
elif choice == '2':
    run_regularized_logistic_regression()
else:
    print("Invalid choice. Please enter 1 or 2.")

# run_logistic_regression()
# run_regularized_logistic_regression()
