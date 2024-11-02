import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * np.sin(3 * x) - np.exp(x)

def newton_raphson(x0):
    x = np.array([x0])
    for j in range(1000):
        f_x = f(x[j])
        df = np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j])
        x = np.append(x, x[j] - f_x / df)
        fc = x[j + 1] * np.sin(3 * x[j + 1]) - np.exp(x[j + 1])
        
        if abs(f_x) < 1e-6:
            break
    return x, j + 1

def bisection(a, b):
    x = np.array([])
    for j in range(1000):
        mid = (a + b) / 2
        x = np.append(x, mid)
        f_mid = f(mid)
        if abs(f_mid) < 1e-6:
            break
        elif f(a) * f_mid < 0:
            b = mid
        else:
            a = mid
    return x, j + 1
    
A1, i = newton_raphson(-1.6)
A2, j = bisection(-0.7, -0.4)
A3 = [i, j]

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = A + B
A5 = (3*x - 4*y).reshape(-1)
A6 = (A @ x).reshape(-1)
A7 = (B @ (x-y)).reshape(-1)
A8 = (D @ x).reshape(-1)
A9 = (D @ y + z).reshape(-1)
A10 = A @ B
A11 = B @ C
A12 = C @ D
A12 = C @ D
