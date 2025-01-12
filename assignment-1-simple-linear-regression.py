import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

independent_file = "./linearX.csv"  
dependent_file = "./linearY.csv"   

X = np.loadtxt(independent_file, delimiter=",") 
Y = np.loadtxt(dependent_file, delimiter=",")    

X = (X - np.mean(X)) / np.std(X)
Y = Y.reshape(-1, 1)

X = (X - np.mean(X)) / np.std(X)

theta0, theta1 = 0, 0
alpha = 0.01  
iterations = 50
m = len(Y)

costs = []
for i in range(iterations):
    predictions = theta0 + theta1 * X
    cost = (1 / (2 * m)) * np.sum((predictions - Y) ** 2)
    costs.append(cost)
    
    temp0 = theta0 - alpha * (1 / m) * np.sum(predictions - Y)
    temp1 = theta1 - alpha * (1 / m) * np.sum((predictions - Y) * X)
    
    theta0, theta1 = temp0, temp1
    
    if len(costs) > 1 and abs(costs[-1] - costs[-2]) < 1e-6:
        break

print(f"Final values: theta0 = {theta0}, theta1 = {theta1}")
print(f"Final cost: {cost}")
print(f"Number of iterations: {len(costs)}")

plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.show()
