# Logistic Regression (Linear + Regularized)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ========== Function Definitions ==========

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X @ theta)
    J = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    grad = (1/m) * (X.T @ (h - y))
    return J, grad


def costFunctionReg(theta, X, y, lambda_):
    m = y.size
    h = sigmoid(X @ theta)
    J = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + (lambda_/(2*m)) * np.sum(np.square(theta[1:]))
    grad = (1/m) * (X.T @ (h - y))
    grad[1:] = grad[1:] + (lambda_/m) * theta[1:]
    return J, grad


def plotData(X, y):
    pos = y == 1
    neg = y == 0
    plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)


def plotDecisionBoundary(theta, X, y, mapFeature=None):
    plotData(X[:, 1:3], y)
    if X.shape[1] <= 3:
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y)
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(np.array([u[i]]), np.array([v[j]])) @ theta
        z = z.T
        plt.contour(u, v, z, levels=[0], linewidths=2)
    plt.xlabel('Exam 1 score' if mapFeature is None else 'Microchip Test 1')
    plt.ylabel('Exam 2 score' if mapFeature is None else 'Microchip Test 2')
    plt.legend()
    plt.show()


def predict(theta, X):
    return sigmoid(X @ theta) >= 0.5


def mapFeature(X1, X2, degree=6):
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, (X1**(i-j) * X2**j).reshape(-1, 1)))
    return out


# ============ Part 1: Load and Plot Data ============
data1 = np.loadtxt('ex2data1.txt', delimiter=',')
X1 = data1[:, 0:2]
y1 = data1[:, 2]
m1 = y1.size
X1 = np.concatenate([np.ones((m1, 1)), X1], axis=1)
initial_theta1 = np.zeros(X1.shape[1])
cost1, grad1 = costFunction(initial_theta1, X1, y1)
res1 = minimize(fun=lambda t: costFunction(t, X1, y1)[0],
                x0=initial_theta1,
                jac=lambda t: costFunction(t, X1, y1)[1],
                method='TNC',
                options={'maxiter': 400})
theta1 = res1.x
plotDecisionBoundary(theta1, X1, y1)
print("Train Accuracy:", np.mean(predict(theta1, X1) == y1) * 100)
print("For a student with scores 45 and 85, predict admission probability:",
      sigmoid(np.dot(np.array([1, 45, 85]), theta1)))


# ============ Part 2: Regularized Logistic Regression ============
data2 = np.loadtxt('ex2data2.txt', delimiter=',')
X2 = data2[:, 0:2]
y2 = data2[:, 2]
X2_mapped = mapFeature(X2[:, 0], X2[:, 1])
initial_theta2 = np.zeros(X2_mapped.shape[1])
lambda_vals = [1, 0, 100]

for lambda_ in lambda_vals:
    cost2, grad2 = costFunctionReg(initial_theta2, X2_mapped, y2, lambda_)
    res2 = minimize(fun=lambda t: costFunctionReg(t, X2_mapped, y2, lambda_)[0],
                    x0=initial_theta2,
                    jac=lambda t: costFunctionReg(t, X2_mapped, y2, lambda_)[1],
                    method='TNC',
                    options={'maxiter': 400})
    theta2 = res2.x
    print(f"Train Accuracy with lambda={lambda_}: {np.mean(predict(theta2, X2_mapped) == y2) * 100:.2f}%")
    plotDecisionBoundary(theta2, X2_mapped, y2, mapFeature)

