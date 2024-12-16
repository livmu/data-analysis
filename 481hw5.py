import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, kron, identity
from scipy.integrate import solve_ivp
from scipy.linalg import lu, solve_triangular
from scipy.fftpack import fft2, ifft2

# Define ODEs
def fft_rhs(t, w, nx, ny, N, kx, ky, k, nu):
    wt = fft2(w.reshape((nx, ny)))
    psi = np.real(ifft2(-wt/k)).reshape(N)
    return (nu * matA.dot(w) + (matB.dot(w)) 
           * (matC.dot(psi)) - (matB.dot(psi)) * (matC.dot(w)))

def ab_rhs(t, w, A, B, C, nu):
    psi = np.linalg.solve(A, w)
    return nu * A.dot(w) + (B.dot(w)) * (C.dot(psi)) - (B.dot(psi)) * (C.dot(w))

def lu_rhs(t, w, A, B, C, nu, P, L, U):
    Pw = np.dot(P, w)
    y = solve_triangular(L, Pw, lower=True)
    psi = solve_triangular(U, y, lower=False)
    return nu * A.dot(w) + (B.dot(w)) * (C.dot(psi)) - (B.dot(psi)) * (C.dot(w))

def span(L, n):
    k = (2 * np.pi / L) * np.concatenate((np.arange(0, n/2), np.arange (-n/2, 0)))
    k[0] = 1e-6
    return k

# Define arrays
m = 64    # N value in x and y directions
n = m * m  # total size of matrix

L = 10; dx = 2 * L / m  # Grid spacing

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

matA = spdiags(diagonals, offsets, n, n).toarray()
matA /= dx**2
matA[0, 0] = 2 / (dx**2)

# Find first derivative wrt x
diagonals = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets = [-(n-m), -m, m, n-m]
matB = spdiags(diagonals, offsets, n, n).toarray()
matB /= (2 * dx)

# Find first derivatice wrt y
diagonals = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [-(m-1), -1, 1, m-1]
matC = spdiags(diagonals, offsets, n, n).toarray()
matC /= (2 * dx)

tspan = np.arange(0, 4.5, 0.5) # Define time span
nu = 0.001

# Parameters
Lx = 20    # spatial domain of x
Ly = 20    # spatial domain of y
nx = 64   # number of discretization points in x
ny = 64   # number of discretization points in y
N = nx * ny   # elements in reshaped initial condition

x2 = np.linspace(-Lx/2, Lx/2, nx+1) # x domain
x = x2[:nx]   
y2 = np.linspace(-Ly/2, Ly/2, ny+1) # y domain
y = y2[:ny]  
X, Y = np.meshgrid(x, y)  # make 2D

U = np.exp(-X**2 - Y**2 / 20) # Generate a Gaussian matrix
u = U.flatten()#[:N].reshape(N, 1) # Reshape into a vector

kx, ky = np.meshgrid(span(Lx, nx), span(Ly, ny))
k = kx**2 + ky**2

### PART A: FFT ###
sol = solve_ivp (
      fft_rhs,
      (tspan[0], tspan[-1]),
      u,
      t_eval=tspan,
      args=(nx, ny, N, kx, ky, k, nu),
      method='RK45'
)

A1 = sol.y

### PART B: A\b ###
sol = solve_ivp (
      ab_rhs,
      (tspan[0], tspan[-1]),
      u,
      t_eval=tspan,
      args=(matA, matB, matC, nu),
      method='RK45'
)

A2 = sol.y

### PART C: LU ###
P, L, U = lu(matA)
sol = solve_ivp (
      lu_rhs,
      (tspan[0], tspan[-1]),
      u,
      t_eval=tspan,
      args=(matA, matB, matC, nu, P, L, U),
      method='RK45'
)

A3 = sol.y