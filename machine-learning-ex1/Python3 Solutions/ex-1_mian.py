### Linear regression with one variable


#  [] warm_up_exercise.py- Simple example function
#  [] plotData.py- Function to display the dataset
#  [] computeCost.py- Function to compute the cost of linear regression
#  [] gradientDescent.py- Function to run gradient descent

import numpy as np
import matplotlib.pyplot as plt
from warmUpExercise import warm_up_exercise
from plotData import plot_data
from computeCost import compute_cost
from gradientDescent import gradient_descent

# Initialization
print('Running warm_up_exercise ...')
print('5x5 Identity Matrix:')
print(warm_up_exercise())

input("Program paused. Press enter to continue.")

# ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

plot_data(X, y)

input("Program paused. Press enter to continue.")

# =================== Part 3: Gradient Descent ===================
print('Running Gradient Descent ...')

X = np.column_stack((np.ones(m), X))  # Add a column of ones
theta = np.zeros(2)

iterations = 1500
alpha = 0.01

cost = compute_cost(X, y, theta)
print(f'Initial Cost: {cost}')

theta = gradient_descent(X, y, theta, alpha, iterations)

print(f'Theta found by gradient descent: {theta[0]} {theta[1]}')

plt.plot(X[:, 1], X @ theta, '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print(f'For population = 35,000, we predict a profit of {predict1 * 10000}')

predict2 = np.dot([1, 7], theta)
print(f'For population = 70,000, we predict a profit of {predict2 * 10000}')

input("Program paused. Press enter to continue.")

# ======= Part 4: Visualizing J(theta_0, theta_1) =======
print('Visualizing J(theta_0, theta_1) ...')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(X, y, t)

J_vals = J_vals.T  # Transpose for correct orientation

# Surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(T0, T1, J_vals, cmap='viridis')
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
plt.show()

# Contour plot
plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
plt.show()
