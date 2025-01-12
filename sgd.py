import numpy as np
import matplotlib.pyplot as plt

Y = np.loadtxt('dependent_variable.txt')
m = len(Y)

X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

X = np.c_[np.ones(m), X]

def compute_cost(X, Y, theta):
    predictions = X @ theta
    cost = (1 / (2 * len(Y))) * np.sum((predictions - Y) ** 2)
    return cost

def gradient_descent(X, Y, theta, alpha, iterations):
    costs = []
    for _ in range(iterations):
        gradients = (1 / m) * X.T @ (X @ theta - Y)
        theta -= alpha * gradients
        costs.append(compute_cost(X, Y, theta))
    return theta, costs

def stochastic_gradient_descent(X, Y, theta, alpha, iterations):
    costs = []
    for _ in range(iterations):
        for i in range(len(Y)):
            xi = X[i, :].reshape(1, -1)
            yi = Y[i]
            gradients = (xi.T @ (xi @ theta - yi)) / m
            theta -= alpha * gradients
            costs.append(compute_cost(X, Y, theta))
    return theta, costs

def mini_batch_gradient_descent(X, Y, theta, alpha, iterations, batch_size):
    costs = []
    for _ in range(iterations):
        for i in range(0, len(Y), batch_size):
            xi = X[i:i + batch_size, :]
            yi = Y[i:i + batch_size]
            gradients = (xi.T @ (xi @ theta - yi)) / batch_size
            theta -= alpha * gradients
        costs.append(compute_cost(X, Y, theta))
    return theta, costs

theta = np.zeros(X.shape[1])
alpha = 0.01
iterations = 50
batch_size = 32

theta_batch, costs_batch = gradient_descent(X, Y, theta.copy(), alpha, iterations)
theta_sgd, costs_sgd = stochastic_gradient_descent(X, Y, theta.copy(), alpha, iterations)
theta_mini_batch, costs_mini_batch = mini_batch_gradient_descent(X, Y, theta.copy(), alpha, iterations, batch_size)

plt.plot(range(len(costs_batch)), costs_batch, label='Batch Gradient Descent')
plt.plot(range(len(costs_sgd)), costs_sgd, label='Stochastic Gradient Descent')
plt.plot(range(len(costs_mini_batch)), costs_mini_batch, label='Mini-Batch Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations')
plt.legend()
plt.show()

print(f"Final cost using Batch Gradient Descent: {costs_batch[-1]}")
print(f"Final cost using Stochastic Gradient Descent: {costs_sgd[-1]}")
print(f"Final cost using Mini-Batch Gradient Descent: {costs_mini_batch[-1]}")
