import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp

# Functions
def tanh(x):
    return np.sinh(x) / np.cosh(x)

def lambda_f(u, v):
    return 1 - (np.abs(u)**2 + np.abs(v)**2)

def omega_f(u, v):
    return -beta * (np.abs(u)**2 + np.abs(v)**2)

# Define parameters
tspan = np.arange(0, 4.5, 0.5)
beta = 1; m = 1; d1 = d2 = 0.1
nu = 0.001; T = 4
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

# Compute u and v
u = tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
v = tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

# Define the ODE system
def spc_rhs(t, wt2):
    utc, vtc = wt2[0:N], wt2[N:]
    ut, vt = utc.reshape((nx, ny)), vtc.reshape((nx, ny))
    u, v = ifft2(ut), ifft2(vt)

    du = fft2(lambda_f(u, v) * u - omega_f(u, v) * v)
    dv = fft2(omega_f(u, v) * u + lambda_f(u, v) * v)

    # Define spectral k values
    kx = 2 * np.pi * np.fft.fftfreq(u.shape[1], d=Lx/nx)
    ky = 2 * np.pi * np.fft.fftfreq(u.shape[0], d=Ly/ny)
    KX, KY = np.meshgrid (kx, ky)
    K = KX ** 2 + KY ** 2
    
    return np.hstack([(-d1*K*ut + du).reshape(N), (-d2*K*vt + dv).reshape(N)])

# Solve the ODE and plot the results
t0 = np.hstack([fft2(u).reshape(N),fft2(v).reshape(N)])
sol = solve_ivp(spc_rhs, [0,T], t0, t_eval=tspan, args=(), method='RK45')
A1 = sol.y
print(A1)
print(A1.shape)

# Chebyshev differentiation matrix
def cheb(N):
	if N==0: 
		D = 0.; x = 1.
	else:
		n = arange(0,N+1)
		x = cos(pi*n/N).reshape(N+1,1) 
		c = (hstack(( [2.], ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = tile(x,(1,N+1))
		dX = X - X.T
		D = dot(c,1./c.T)/(dX+eye(N+1))
		D -= diag(sum(D.T,axis=0))
	return D, x.reshape(N+1)

N = 30; N2 = (N+1)**2
D, x = cheb(N)
D[N, :] = 0
D[0, :] = 0
D2 = np.dot(D, D) / 100 # Second derivative matrix
y = x

I = np.eye(len(D2))
L = kron(I, D2) + kron(D2, I)  # 2D Laplacian

X, Y = np.meshgrid(x, y)
X *= 10; Y *= 10

# Compute u and v
u = tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
v = tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

# Define the ODE system
def rhs(t, wt2):
    ut, vt = wt2[0:N2], wt2[N2:]

    du = lambda_f(ut, vt) * ut - omega_f(ut, vt) * vt
    dv = omega_f(ut, vt) * ut + lambda_f(ut, vt) * vt

    return np.hstack([(d1*(L@ut) + du).reshape(N2), (d2*(L@vt) + dv).reshape(N2)])

# Solve the ODE and plot the results
t0 = np.hstack([u.reshape(N2), v.reshape(N2)])
sol = solve_ivp(rhs, [0,T], t0, t_eval=tspan, args=(), method='RK45')
A2 = sol.y
print(A2)
print(A2.shape)