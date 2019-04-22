#!/usr/bin/env python
# coding: utf-8

import numpy as np

# Jacobi, Gauss-Seidel and SOR methods for 1d FDM for Poisson's equation

def jacobi(v, b):
    v[1:-1] = 0.5 * (b[1:-1] + v[:-2] + v[2:])

def gauss_seidel(v, b):
    for i in range(1, len(v) - 1):
        v[i] = 0.5 * (b[i] + v[i - 1] + v[i + 1])

def sor(v, b, omega):
    for i in range(1, len(v) - 1):
        v[i] = (1 - omega) * v[i] + omega * 0.5 * (b[i] + v[i - 1] + v[i + 1])

# What follows is plotting and testing code

N = 64

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

import matplotlib.pyplot as plt

# # Direct solver: Gaussian elimation

# Before considering iterative solver, let us first think about the basic direct method: _Gaussian elimination_. The matrix of the Laplacian is very simple, it is a [tridiagonal matrix](https://en.wikipedia.org/wiki/Tridiagonal_matrix)
# and therefore $Ax = b$ can be effiently solved by the [tridiagonal matrix algorithm](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm).
#
# Parameters and the right hand side $b$. The matrix multiplication is written explicitly, we don't want to store the matrix.

def tridiag_solver(b):
    """Tridiagonal matrix solver for the discrete Laplacian matrix, assuming the Dirichlet
    boundary condition x[0] = x[-1] = 0."""
    b = np.copy(b)
    x = np.zeros_like(b)
    c = np.zeros_like(b)

    for i in range(1, len(x) - 1):
        c[i] = -1. / (2 +  c[i - 1])
        b[i] = (b[i] + b[i - 1]) / (2 + c[i - 1])

    for i in reversed(range(1, len(x) - 1)):
        x[i] = b[i] - c[i] * x[i + 1]

    return x

def Amul(x):
    """ Multiplication by the matrix A. """
    return np.pad(-x[:-2] + 2. * x[1:-1] - x[2:], 1, 'constant')

def parameters(N):
    """ Initialize interesting test parameters. """
    h = 1. / N
    x = np.linspace(0., 1., N + 1)
    b = h**2 * (3. * (2 * np.cos(5.* np.pi * x) + 3* np.sin(2.* np.pi * x) + 0.5 * np.sin(np.pi * x)))
    b[0] = b[-1] = 0
    v_star = tridiag_solver(b) # exact solution of the fdm scheme

    return x, v_star, b

x, v_star, b  = parameters(N)

plt.figure(figsize = (16, 5))
plt.subplot(121, title='$u^*$')
plt.plot(x, v_star)
plt.subplot(122, title='$b$')
plt.plot(x[1:-1], b[1:-1])
plt.savefig('fdm_data.png')
plt.clf()


def trace(stepper, v0, b, steps):
    """ Trace a solution. """
    v = np.zeros_like(v0)
    N = len(v)
    for i in range(steps):
        stepper(v, b)
        yield np.copy(v)

def plot_steps(vs, v_star, b, x, title, filename):
    plt.figure(figsize=(16,6))
    plt.suptitle(title)
    ax_v = plt.subplot(121)
    ax_v.set_title('$v$')
    ax_v.plot(x, v_star, '--')
    ax_r = plt.subplot(122)
    ax_r.set_title(r'$r$')
    for v in vs:
        r = b - Amul(v)

        ax_v.plot(x, v)
        ax_r.plot(x, r)

    plt.savefig(filename)
    plt.clf()

def make_sol_animation(x, v_star, b, vs, label):
    from matplotlib import animation

    rs = [[b - Amul(v) for v in vs] for vs in vs]

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(121, autoscale_on=False,
                         xlim=(np.min(x), np.max(x)), ylim=(min(map(np.min,vs)), max(map(np.max,vs))), title=r'$v$')
    ax.plot(x, v_star, '--')
    vplot = [ax.plot([], [], label = l)[0] for v, l in zip(vs, label)]
    ax.legend(loc = 'upper right')

    ax = fig.add_subplot(122, autoscale_on=False,
                         xlim=(np.min(x), np.max(x)), ylim=(min(map(np.min,rs)), max(map(np.max,rs))), title=r'$r$')
    ax.plot(x, b - Amul(v_star))
    rplot = [ax.plot([], [], label = l)[0] for v, l in zip(vs, label)]
    ax.legend(loc = 'upper right')

    def init():
        return vplot + rplot

    def animate(i):
        for p, v in zip(vplot, vs):
            if i < len(v):
                p.set_data(x, v[i])
        for p, r in zip(rplot, rs):
            if i < len(r):
                p.set_data(x, r[i])
        return vplot + rplot

    ani = animation.FuncAnimation(fig, animate, frames=len(vs[0]),
                                  interval = 50, blit=True, init_func=init)

    ani.save('fdm-1d.mp4')


x, v_star, b = parameters(N)

print('Computing...')

jac_us = list(trace(jacobi, v_star, b, N))
gs_us = list(trace(gauss_seidel, v_star, b, N))
omega = 2. / (1. + np.sin(np.pi / N))
sor_us = list(trace(lambda v, b: sor(v, b, omega), v_star, b, N))

print('Plotting...')
plot_steps(jac_us, v_star, b, x, 'Jacobi', 'jacobi.png')
plot_steps(gs_us, v_star, b, x, 'Gauss-Seidel', 'gauss_seidel.png')
plot_steps(sor_us, v_star, b, x, 'SOR $\\omega = {}$'.format(omega), 'sor.png')

print('Animating...')
make_sol_animation(x, v_star, b, [jac_us, gs_us, sor_us], ['Jacobi', 'Gauss-Seidel', 'SOR $\\omega = {:.4}$'.format(omega)])


def make_error_plot(v_star, b, vs, label):
    rs = [[np.sqrt(np.sum((b - Amul(v))**2)) for v in vs] for vs in vs]

    fig = plt.figure(figsize=(12,5))
    for s, l in zip(rs, label):
        plt.semilogy(range(len(s)), s, label = l, linewidth = 3.)

    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel(r'$||r||$')
    plt.savefig('fdm-1d-error.png')
    plt.clf()

print('Making error plot...')
make_error_plot(v_star, b, [jac_us, gs_us, sor_us], ['Jacobi', 'Gauss-Seidel', 'SOR $\\omega = {:.4}$'.format(omega)])
#
#
