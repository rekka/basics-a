# Simple implementation of a multigrid method for the FDM discretization of the Poisson equation
# -u'' = b
import matplotlib.pyplot as plt
import numpy as np

def damped_jacobi(v, b, omega):
    v[1:-1] = (1. - omega) * v[1:-1] + omega * 0.5 * (b[1:-1] + v[:-2] + v[2:])

def jacobi23(v, b):
    damped_jacobi(v, b, 2. / 3.)


def restrict(v):
    """ Restriction operator. """
    return np.pad(0.25 * v[1:-3:2] + 0.5 * v[2:-2:2] + 0.25 * v[3:-1:2], 1, 'constant')

def prolong(v):
    """ Prolongation operator. """
    r = np.zeros(2 * len(v) - 1)
    r[::2] = v
    r[1::2] = 0.5 * (v[:-1] + v[1:])
    return r


def Amul(v):
    """ Multiplication by a dicrete Laplacian matrix. """
    return np.pad(-v[:-2] + 2. * v[1:-1] - v[2:], 1, 'constant')

def vcycle(v, b):
    """ One vcycle of a multigrid method. """
    if (len(v) - 1) & (len(v) - 2) != 0:
        raise ValueError("Lenth of v must be 2**n + 1.")

    for i in range(3):
        jacobi23(v, b)

    if len(v) <= 3:
        return

    r = b - Amul(v)
    r2 = 4. * restrict(r)
    e2 = np.zeros_like(r2)
    vcycle(e2, r2)
    v += prolong(e2)

    for i in range(3):
        jacobi23(v, b)

# Testing

def tridiag_solver(b):
    """Tridiagonal matrix solver for the discrete Laplacian matrix, assuming the Dirichlet
    boundary condition x[0] = x[-1] = 0."""
    b = np.copy(b)
    v = np.zeros_like(b)
    c = np.zeros_like(b)

    for i in range(1, len(v) - 1):
        c[i] = -1. / (2 +  c[i - 1])
        b[i] = (b[i] + b[i - 1]) / (2 + c[i - 1])

    for i in reversed(range(1, len(v) - 1)):
        v[i] = b[i] - c[i] * v[i + 1]

    return v

def parameters(N):
    """ Initialize some test data. """
    h = 1. / N
    x = np.linspace(0., 1., N + 1)
    b = h**2 * (3. * (2 * np.sin(5.* np.pi * x) + 3* np.sin(2.* np.pi * x) + 0.5 * np.sin(np.pi * x))) * x**2
    b[0] = b[-1] = 0

    # "Exact" solution of the linear system.
    v_star = tridiag_solver(b)

    return x, b, v_star

N = 2**20 # Resolution of the test problem
x, b, v_star = parameters(N) # Initialize the test parameters

fig = plt.figure(figsize=(16, 6))
fig.suptitle('$N = {}$'.format(N))
plt.subplot(121, title = '$r$')

v = np.zeros_like(x)

r = b - Amul(v)
plt.plot(x, r, label = 'k = 0')
r_norm0 = np.sqrt(r.dot(r))
r_norm = [r_norm0]

for i in range(8):
    vcycle(v, b)

    r = b - Amul(v)
    r_norm.append(np.sqrt(r.dot(r)))
    plt.plot(x, r)


plt.legend()
plt.subplot(122)
plt.semilogy(range(0, len(r_norm)), r_norm / r_norm0, label = r'$||r_k|| / ||r_0||$')
plt.semilogy(range(0, len(r_norm)), [10**(-k) for k in range(len(r_norm))], '--', label = r'$10^{-k}$')
plt.legend()
plt.xlabel('V-cycle iteration')


plt.show()


