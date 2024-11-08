import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp, simpson, solve_ivp, RK45
from scipy.optimize import bisect
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

### PART A ###
def shoot2(x, y, beta):
    return [y[1], (x**2 - beta) * y[0]]

# Normalize the eigenfunction
def normalize(phi):
    norm = np.sqrt(np.trapz(phi**2, xshoot, axis=0))
    return np.abs(phi / norm)

def get_eigenvalue(n):
    beta = n # initial value of eigenvalue beta
    dbeta = 0.1 # default step size in beta
    for _ in range(1000):  # begin convergence loop for beta
        A = np.sqrt(K * L**2 - beta); x0 = [1, A]
        sol = solve_ivp(shoot2, [-L, L + 0.1], x0, args=(beta,), t_eval=xshoot)
        y1 = sol.y[0,-1]; y2 = sol.y[1,-1]
        
        if abs(y2 + A*y1) < tol: # check for convergence
            return beta, sol  # get out of convergence loop
        
        if ((-1) ** (modes + 1) * (y2 + A * y1)) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta = dbeta / 2
    
    return beta, sol

tol = 1e-6  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
n0 = 0.1; L = 4; K = 1; xshoot = np.linspace(-L, L, 81)  # Discretized domain

eigenvalues = []
eigenfunctions = []

beta_start = n0  # beginning value of beta
for modes in range(1, 6):  # begin mode loop
    beta, sol = get_eigenvalue(beta_start)
    beta_start = beta + 0.1 # after finding eigenvalue, pick new start
    y = normalize(sol.y[0])
    
    eigenvalues.append(beta)  # Store the eigenvalue
    eigenfunctions.append(y)  # Store the normalized eigenfunction
    plt.plot(xshoot, y, col[modes - 1])  # Plot the eigenfunction

plt.show()  # end mode loop
A1 = np.column_stack(eigenfunctions)
A2 = np.array(eigenvalues)

### PART B ###
dx = 0.1; L = 4; K = 1; xshoot = np.arange(-L, L + dx, dx)  # parameters

# construct matrix
main_diag = -(2 / dx**2) - K * xshoot[1:-1]**2 * np.ones(len(xshoot) - 2)
upper_diag = 1 * np.ones(len(xshoot) - 2) / dx**2
lower_diag = 1 * np.ones(len(xshoot) - 2) / dx**2

main_diag[0] += 4/(3*dx**2)
main_diag[-1] += 4/(3*dx**2)
upper_diag[0] -= 1/(3*dx**2)
lower_diag[-2] -= 1/(3*dx**2)

A = diags([main_diag, lower_diag, upper_diag], [0, -1, 1]).toarray()

# solve for the first five eigenvalues and eigenfunctions
eigenvalues, eigenfunctions = eigs(A, k=5, which='SM')  # sort small to large

left = 4/3 * eigenfunctions[0, :] - 1/3 * eigenfunctions[1, :]
right = -1/3 * eigenfunctions[-2, :] + 4/3 * eigenfunctions[-1, :]
eigenfunctions = np.vstack((left, eigenfunctions, right))

# normalize
eigenvalues = eigenvalues.real
eigenfunctions = normalize(eigenfunctions.real)

for i in range(len(eigenvalues)):
    plt.plot(xshoot, np.abs(eigenfunctions[:, i]), col[i])

plt.show()

A3 = abs(eigenfunctions)
A4 = abs(eigenvalues)

### PART C ###
def shoot2(x, y, beta, gamma):
    return [y[1], (gamma * y[0]**2 + K * x**2 - beta) * y[0]]

def get_eigenvalue(n, L, K, A, gamma, modes, xshoot):
    dA = 0.01
    for _ in range(100):
        beta = n # initial value of eigenvalue beta
        dbeta = 0.1 # default step size in beta
        for _ in range(100):
            diff = np.sqrt(K * L**2 - beta); y = [A, diff * A]
            sol = solve_ivp(shoot2, [xshoot[0], xshoot[-1]], y, args=(beta,gamma,), t_eval=xshoot)
            y1 = sol.y[0,-1]; y2 = sol.y[1,-1]
            boundary = y2 + diff * y1

            if abs(boundary) < tol: # check for convergence
                break

            if ((-1) ** (modes + 1) * boundary) > 0:
                beta += dbeta
            else:
                beta -= dbeta
                dbeta = dbeta / 2
        
        # Check whether it is focused
        integral = simpson(sol.y[0] ** 2, x=sol.t)
        if abs(integral - 1) < tol:
            break

        # Adjust to steps of A
        if integral < 1:
            A += dA
        else:
            A -= dA
            dA /= 2
    
    return beta, abs(sol.y[0])

def find_eigenfunctions(n0, L, K, A, gamma, xshoot):
    eigenvalues = []
    eigenfunctions = []
    
    beta_start = n0  # beginning value of beta
    for modes in range(1, 3):  # begin mode loop
        beta, sol = get_eigenvalue(beta_start, L, K, A, gamma, modes, xshoot)
        beta_start = beta + 0.2 # after finding eigenvalue, pick new start
        
        eigenvalues.append(beta)  # Store the eigenvalue
        eigenfunctions.append(sol)  # Store the normalized eigenfunction
    
    return np.column_stack(eigenfunctions), np.array(eigenvalues)

L = 2; dx = 0.1; xshoot = np.arange(-L, L + dx, dx)
A5, A6 = find_eigenfunctions(n0=0.1, L=2, K=1, A=1e-6, gamma=0.05, xshoot=xshoot)
A7, A8 = find_eigenfunctions(n0=0.1, L=2, K=1, A=1e-6, gamma=-0.05, xshoot=xshoot)

### PART D ###

# Define the ODE function
def hw1_rhs_a(x, y, beta):
    return [y[1], (x**2 - beta) * y[0]]
    
# Parameters
beta = 1; gamma = 0; L = 2; x_span = [-L, L]
y0 = [1, np.sqrt(L**2 - beta)]
tolerances = [10**-i for i in range(4, 11)]

methods = ['RK45', 'RK23', 'Radau', 'BDF']
slopes = []

# Solve the ODE for each method and tolerance
for method in methods:
    step_sizes = []
    for tol in tolerances:
        options = {'rtol': tol, 'atol': tol}
        sol = solve_ivp(hw1_rhs_a, x_span, y0, method=method, args=(beta,), **options)
        step_sizes.append(np.mean(np.diff(sol.t)))
    
    # Log-log plot and polyfit
    log_tol = np.log10(tolerances)
    log_step = np.log10(step_sizes)
    slope, _ = np.polyfit(log_step, log_tol, 1)
    slopes.append(slope)
    
    plt.plot(log_step, log_tol, label=f'{method}')

# Plotting
plt.legend()
plt.grid()
plt.show()

A9 = np.array(slopes) # save the slopes

### PART E ###
def percent_error(epsilon_num, epsilon_exact):
    return 100 * np.abs(epsilon_num - epsilon_exact) / epsilon_exact

def eigenfunction_error(phi_num, phi_exact, x):
    f = np.abs(phi_num) - np.abs(phi_exact)
    return np.trapz(f**2, x=x)

def get_phi(n, x, hermite_eqs):
    return np.exp(-1 * x**2 / 2) * hermite_eqs[:,n] / np.sqrt(factorial(n) * 2**n * np.sqrt(np.pi))
    
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def hermite(x, n):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return -2 + 4 * x**2
    elif n == 3:
        return -12 * x + 8 * x**3
    elif n == 4:
        return 12 - 48 * x**2 + 16 * x**4
    return None

L = 4; dx = 0.1; xshoot = np.arange(-L, L + dx, dx)
hermite_eqs = np.column_stack([hermite(xshoot, 0), hermite(xshoot, 1), hermite(xshoot, 2), 
                               hermite(xshoot, 3), hermite(xshoot, 4)])

# Initialize error arrays
A10 = []; A11 = []; A12 = []; A13 = []
phi = np.zeros(hermite_eqs.shape)
                 
for n in range(5):
    phi[:,n] = get_phi(n, xshoot, hermite_eqs)
    phi_exact = phi[:,n]
    eigen_exact = 2*n + 1

    A10.append(eigenfunction_error(A1.T[n], phi.T[n], xshoot))
    A11.append(percent_error(A2[n], eigen_exact))
    
    A12.append(eigenfunction_error(A3.T[n], phi.T[n], xshoot))
    A13.append(percent_error(A4[n], eigen_exact))