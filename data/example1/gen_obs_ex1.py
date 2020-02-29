import numpy as np
import csv

def f(x, k):
    mu = x / 2.0 + (25 * x) / (1 + x**2) + 8 * np.cos(1.2*k)
    sigma = 10;
    return np.random.normal(mu, sigma, 1)

def obs(x):
    mu = x**2 / 20.0
    sigma = 1
    return np.random.normal(mu, sigma, 1)

vals = np.array(range(0,100))
x = np.empty(vals.size) # exact values
for index, k in enumerate(vals):
    if index == 0:
        x[index] = np.random.normal(0, 0.5, 1)
    else:
        x[index] = f(x[index-1], k)

o = np.empty(vals.size) # observations
for index, exact in enumerate(x):
    o[index] = obs(x[index])


with open('exact_ex1.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    for index, time in enumerate(x):
        writer.writerow([vals[index], x[index]])

with open('obs_ex1.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    for index, time in enumerate(o):
        writer.writerow([vals[index], o[index]])

print(x)        
print(o)
