import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
from scipy.optimize import bisect
from scipy.integrate import solve_ivp

# define the function
def shoot2(x, dummy, beta):
    return [x[1], (dummy**2 - beta) * x[0]]

# normalize the eigenfunction
def normalize(phi):
    norm = np.sqrt(np.trapz(phi**2, xshoot))
    return np.abs(phi / norm)

def get_eigenvalue(n):
    beta = n # initial value of eigenvalue beta
    dbeta = 0.1 # default step size in beta
    for _ in range(1000):  # begin convergence loop for beta
        A = np.sqrt(L**2 - beta); x0 = [1, A]
        y = odeint(shoot2, x0, xshoot, args=(beta,))
        y1 = y[-1,0]; y2 = y[-1,1]
        
        if abs(y2 + A*y1) < tol: # check for convergence
            return beta, y  # get out of convergence loop
        
        if ((-1) ** (modes + 1) * (y2 + A * y1)) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta = dbeta / 2
    
    return beta, y

tol = 1e-6  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
n0 = 0.1; L = 4; xshoot = np.linspace(-L, L, 81)  # Discretized domain

eigenvalues = []
eigenfunctions = []

beta_start = n0  # beginning value of beta
for modes in range(1, 6):  # begin mode loop
    beta, y = get_eigenvalue(beta_start)
    beta_start = beta + 0.1 # after finding eigenvalue, pick new start
    y = normalize(y[:,0])
    
    eigenvalues.append(beta)  # store the eigenvalue
    eigenfunctions.append(y)  # store the normalized eigenfunction

A1 = np.column_stack(eigenfunctions)
A2 = np.array(eigenvalues)