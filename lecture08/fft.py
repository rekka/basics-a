def eigenvector_coeff(u):
    """Coordinates of u in the basis of eigenvectors of the Jacobi iteration matrix:
       w_{m,i} = sin(m i pi / N)
       Assumes u[0] == u[N] == 0."""

    N = len(u) - 1            # u stores u_0, ..., u_N
    a = np.zeros_like(u)      # allocated storage
    i = np.arange(0, len(u))
    for m in range(0, N):
        a[m] = np.dot(u, np.sin(m * i * np.pi / N)) * 2 / N

    return a

# `eigenvector_coeff` can be more efficiently computed using the Fast Fourier Tranform.
# This takes O(N log(N)) time, while eigenvector_coeff takes O(N^2) time.
def sin_fft(u):
    """Sine FFT of u. Assumes u[0] == u[-1] == 0."""
    return -np.imag(np.fft.fft(np.concatenate((u[:-1], -u[:0:-1]))))[:len(u) - 1] * (1. / (len(u) - 1))

# Make a nice plot of the spectrum (coefficients in the eigenvector basis)
def plot_spectrum(data, labels = None):
    ax = plt.axes()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0.))
    ax.spines['right'].set_color('none')
    ax.set_xlabel('$m$')
    ax.set_ylabel('$a_m$')

    for i, d in enumerate(data):
        if labels:
            label = labels[i]
        else:
            label = None
        ax.plot(eigenvector_coeff(d), 'o', label=label)

    if labels:
        ax.legend()

